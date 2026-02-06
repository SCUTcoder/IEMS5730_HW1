package edu.cuhk.iems5730;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.*;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.*;
import java.util.*;

/**
 * Task B: Top-K (default K=3) most similar companies for each company.
 * Similarity = |Common Suppliers| / |Union Suppliers|  (Jaccard)
 *
 * Pipeline:
 *  Job1: company -> sorted unique supplier list (csv) + supplier_count
 *  Job2: supplier -> companies, generate (companyA,companyB) -> commonCount
 *  Job3: compute similarity using counts, emit per company, reducer keeps topK (no common list yet)
 *  Job4: enrich topK pairs with common supplier list by intersecting supplier sets from Job1 output
 */
public class TaskB {

    // -------------------- Utils --------------------

    private static String normalizePair(String a, String b) {
        return (a.compareTo(b) <= 0) ? (a + "\t" + b) : (b + "\t" + a);
    }

    private static Set<String> parseSuppliersCsv(String csv) {
        Set<String> s = new HashSet<>();
        if (csv == null || csv.isEmpty()) return s;
        String[] arr = csv.split(",");
        for (String x : arr) {
            String t = x.trim();
            if (!t.isEmpty()) s.add(t);
        }
        return s;
    }

    private static Map<String, Integer> loadCompanyCounts(Configuration conf, String hdfsPath) throws IOException {
        Map<String, Integer> map = new HashMap<>();
        FileSystem fs = FileSystem.get(conf);
        Path p = new Path(hdfsPath);

        // 允许传目录：读取 part-*
        FileStatus[] stats = fs.isDirectory(p) ? fs.listStatus(p) : new FileStatus[]{fs.getFileStatus(p)};
        for (FileStatus st : stats) {
            String name = st.getPath().getName();
            if (fs.isDirectory(st.getPath())) continue;
            if (!name.startsWith("part-")) continue;

            try (FSDataInputStream in = fs.open(st.getPath());
                 BufferedReader br = new BufferedReader(new InputStreamReader(in))) {
                String line;
                while ((line = br.readLine()) != null) {
                    // format: company \t supplierCsv \t count
                    String[] parts = line.split("\t");
                    if (parts.length >= 3) {
                        String company = parts[0].trim();
                        int cnt = Integer.parseInt(parts[2].trim());
                        map.put(company, cnt);
                    }
                }
            }
        }
        return map;
    }

    private static Map<String, Set<String>> loadCompanySuppliers(Configuration conf, String hdfsPath) throws IOException {
        Map<String, Set<String>> map = new HashMap<>();
        FileSystem fs = FileSystem.get(conf);
        Path p = new Path(hdfsPath);

        FileStatus[] stats = fs.isDirectory(p) ? fs.listStatus(p) : new FileStatus[]{fs.getFileStatus(p)};
        for (FileStatus st : stats) {
            String name = st.getPath().getName();
            if (fs.isDirectory(st.getPath())) continue;
            if (!name.startsWith("part-")) continue;

            try (FSDataInputStream in = fs.open(st.getPath());
                 BufferedReader br = new BufferedReader(new InputStreamReader(in))) {
                String line;
                while ((line = br.readLine()) != null) {
                    // company \t supplierCsv \t count
                    String[] parts = line.split("\t");
                    if (parts.length >= 2) {
                        String company = parts[0].trim();
                        String csv = parts[1].trim();
                        map.put(company, parseSuppliersCsv(csv));
                    }
                }
            }
        }
        return map;
    }

    // -------------------- Job1: Build supplier list + count --------------------

    public static class SupplierListMapper extends Mapper<LongWritable, Text, Text, Text> {
        private final Text outK = new Text();
        private final Text outV = new Text();

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString().trim();
            if (line.isEmpty()) return;
            String[] parts = line.split("\\s+");
            if (parts.length < 2) return;
            outK.set(parts[0]);
            outV.set(parts[1]);
            context.write(outK, outV);
        }
    }

    public static class SupplierListReducer extends Reducer<Text, Text, Text, Text> {
        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            TreeSet<String> suppliers = new TreeSet<>();
            for (Text v : values) suppliers.add(v.toString().trim());
            if (suppliers.isEmpty()) return;

            String csv = String.join(",", suppliers);
            int cnt = suppliers.size();

            // output: company \t supplierCsv \t count
            context.write(key, new Text(csv + "\t" + cnt));
        }
    }

    // -------------------- Job2: Invert supplier->companies and count common suppliers per pair --------------------

    public static class InvertIndexMapper extends Mapper<LongWritable, Text, Text, Text> {
        private final Text outK = new Text();
        private final Text outV = new Text();

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            // input: company \t supplierCsv \t count
            String line = value.toString().trim();
            if (line.isEmpty()) return;
            String[] parts = line.split("\t");
            if (parts.length < 2) return;

            String company = parts[0].trim();
            String supplierCsv = parts[1].trim();
            if (company.isEmpty() || supplierCsv.isEmpty()) return;

            String[] suppliers = supplierCsv.split(",");
            for (String s : suppliers) {
                String sup = s.trim();
                if (sup.isEmpty()) continue;
                outK.set(sup);
                outV.set(company);
                context.write(outK, outV);
            }
        }
    }

    public static class PairCountReducer extends Reducer<Text, Text, Text, IntWritable> {
        private final IntWritable one = new IntWritable(1);
        private final Text outK = new Text();

        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            // supplier -> list of companies that have it
            ArrayList<String> companies = new ArrayList<>();
            for (Text v : values) {
                String c = v.toString().trim();
                if (!c.isEmpty()) companies.add(c);
            }
            if (companies.size() < 2) return;

            Collections.sort(companies);
            // generate all pairs within this supplier
            for (int i = 0; i < companies.size(); i++) {
                for (int j = i + 1; j < companies.size(); j++) {
                    outK.set(companies.get(i) + "\t" + companies.get(j)); // pairKey
                    context.write(outK, one); // each supplier contributes +1 common
                }
            }
        }
    }

    public static class SumCombiner extends Reducer<Text, IntWritable, Text, IntWritable> {
        private final IntWritable outV = new IntWritable();

        @Override
        protected void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable v : values) sum += v.get();
            outV.set(sum);
            context.write(key, outV);
        }
    }

    public static class SumReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        private final IntWritable outV = new IntWritable();

        @Override
        protected void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable v : values) sum += v.get();
            outV.set(sum);
            context.write(key, outV); // pairKey \t commonCount
        }
    }

    // -------------------- Job3: Compute Jaccard and TopK per company (no common list yet) --------------------

    public static class SimilarityEmitMapper extends Mapper<LongWritable, Text, Text, Text> {
        private Map<String, Integer> companyCnt;
        private final Text outK = new Text();
        private final Text outV = new Text();

        @Override
        protected void setup(Context context) throws IOException {
            Configuration conf = context.getConfiguration();
            String job1Out = conf.get("taskb.job1.output");
            if (job1Out == null) throw new IOException("Missing conf: taskb.job1.output");
            companyCnt = loadCompanyCounts(conf, job1Out);
        }

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            // input: companyA \t companyB \t commonCount
            String line = value.toString().trim();
            if (line.isEmpty()) return;
            String[] parts = line.split("\t");
            if (parts.length < 3) return;

            String a = parts[0].trim();
            String b = parts[1].trim();
            int common = Integer.parseInt(parts[2].trim());

            Integer ca = companyCnt.get(a);
            Integer cb = companyCnt.get(b);
            if (ca == null || cb == null) return;

            int union = ca + cb - common;
            if (union <= 0) return;

            double sim = (double) common / (double) union;
            if (sim <= 0) return;

            String simStr = String.format("%.6f", sim);

            // emit to both sides for topK selection
            outK.set(a);
            outV.set(b + "\t" + common + "\t" + simStr);
            context.write(outK, outV);

            outK.set(b);
            outV.set(a + "\t" + common + "\t" + simStr);
            context.write(outK, outV);
        }
    }

    private static class TopRec {
        String other;
        int common;
        double sim;

        TopRec(String other, int common, double sim) {
            this.other = other;
            this.common = common;
            this.sim = sim;
        }
    }

    public static class TopKReducer extends Reducer<Text, Text, Text, Text> {
        private int K;

        @Override
        protected void setup(Context context) {
            K = context.getConfiguration().getInt("topk.k", 3);
        }

        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            // min-heap by sim, tie by other id descending (so smaller id kept)
            PriorityQueue<TopRec> pq = new PriorityQueue<>(K + 1, (x, y) -> {
                int c = Double.compare(x.sim, y.sim);
                if (c != 0) return c;
                return y.other.compareTo(x.other);
            });

            for (Text v : values) {
                String[] p = v.toString().split("\t");
                if (p.length < 3) continue;
                String other = p[0].trim();
                int common = Integer.parseInt(p[1].trim());
                double sim = Double.parseDouble(p[2].trim());

                pq.offer(new TopRec(other, common, sim));
                if (pq.size() > K) pq.poll();
            }

            ArrayList<TopRec> res = new ArrayList<>(pq);
            res.sort((a, b) -> {
                int c = Double.compare(b.sim, a.sim);
                if (c != 0) return c;
                return a.other.compareTo(b.other);
            });

            // output: company \t other \t commonCount \t sim
            for (TopRec r : res) {
                context.write(key, new Text(r.other + "\t" + r.common + "\t" + String.format("%.6f", r.sim)));
            }
        }
    }

    // -------------------- Job4: Enrich with common supplier list --------------------

    public static class EnrichMapper extends Mapper<LongWritable, Text, Text, Text> {
        private final Text outK = new Text();
        private final Text outV = new Text();

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            // input from Job3: company \t other \t commonCount \t sim
            String line = value.toString().trim();
            if (line.isEmpty()) return;
            String[] p = line.split("\t");
            if (p.length < 4) return;

            String company = p[0].trim();
            String other = p[1].trim();
            String common = p[2].trim();
            String sim = p[3].trim();

            outK.set(company);
            outV.set(other + "\t" + common + "\t" + sim);
            context.write(outK, outV);
        }
    }

    public static class EnrichReducer extends Reducer<Text, Text, Text, Text> {
        private Map<String, Set<String>> supplierMap;

        @Override
        protected void setup(Context context) throws IOException {
            Configuration conf = context.getConfiguration();
            String job1Out = conf.get("taskb.job1.output");
            if (job1Out == null) throw new IOException("Missing conf: taskb.job1.output");
            supplierMap = loadCompanySuppliers(conf, job1Out);
        }

        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            String company = key.toString();
            Set<String> s1 = supplierMap.get(company);
            if (s1 == null) s1 = Collections.emptySet();

            for (Text v : values) {
                String[] p = v.toString().split("\t");
                if (p.length < 3) continue;

                String other = p[0].trim();
                int commonCnt = Integer.parseInt(p[1].trim());
                String sim = p[2].trim();

                Set<String> s2 = supplierMap.get(other);
                if (s2 == null) s2 = Collections.emptySet();

                // intersection for topK only -> cheap
                HashSet<String> common = new HashSet<>(s1);
                common.retainAll(s2);

                ArrayList<String> list = new ArrayList<>(common);
                Collections.sort(list);
                String commonStr = "{" + String.join(",", list) + "}";

                // output format (you can adjust to spec):
                // company:other, {commonSuppliers}, similarity
                String outLine = company + ":" + other + ", " + commonStr + ", " + sim;

                // sanity: if spec wants commonCnt, you can append it too.
                // String outLine = company + "\t" + other + "\t" + commonCnt + "\t" + commonStr + "\t" + sim;

                context.write(new Text(outLine), new Text(""));
            }
        }
    }

    // -------------------- Main --------------------

    public static void main(String[] args) throws Exception {
        if (args.length < 2 || args.length > 3) {
            System.err.println("Usage: TaskB <input path> <output path> [K]");
            System.exit(1);
        }

        Configuration conf = new Configuration();
        int K = 3;
        if (args.length == 3) K = Integer.parseInt(args[2]);
        conf.setInt("topk.k", K);

        String input = args[0];
        String output = args[1];

        Path job1Out = new Path(output + "_job1");
        Path job2Out = new Path(output + "_job2");
        Path job3Out = new Path(output + "_job3");
        Path finalOut = new Path(output);

        FileSystem fs = FileSystem.get(conf);
        fs.delete(job1Out, true);
        fs.delete(job2Out, true);
        fs.delete(job3Out, true);
        fs.delete(finalOut, true);

        // ---- Job1
        Job job1 = Job.getInstance(conf, "TaskB-Job1-BuildSupplierLists");
        job1.setJarByClass(TaskB.class);
        job1.setMapperClass(SupplierListMapper.class);
        job1.setReducerClass(SupplierListReducer.class);
        job1.setOutputKeyClass(Text.class);
        job1.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job1, new Path(input));
        FileOutputFormat.setOutputPath(job1, job1Out);
        if (!job1.waitForCompletion(true)) System.exit(1);

        // ---- Job2
        Job job2 = Job.getInstance(conf, "TaskB-Job2-CountCommonSuppliers");
        job2.setJarByClass(TaskB.class);
        job2.setMapperClass(InvertIndexMapper.class);
        job2.setReducerClass(PairCountReducer.class);
        job2.setMapOutputKeyClass(Text.class);
        job2.setMapOutputValueClass(Text.class);
        job2.setOutputKeyClass(Text.class);
        job2.setOutputValueClass(IntWritable.class);

        // combiner + reducer sum commonCount per pair
        job2.setCombinerClass(SumCombiner.class);
        job2.setReducerClass(SumReducer.class);

        FileInputFormat.addInputPath(job2, job1Out);
        FileOutputFormat.setOutputPath(job2, job2Out);
        if (!job2.waitForCompletion(true)) System.exit(1);

        // ---- Job3
        Configuration conf3 = new Configuration(conf);
        conf3.set("taskb.job1.output", job1Out.toString());

        Job job3 = Job.getInstance(conf3, "TaskB-Job3-TopKByJaccard");
        job3.setJarByClass(TaskB.class);
        job3.setMapperClass(SimilarityEmitMapper.class);
        job3.setReducerClass(TopKReducer.class);
        job3.setOutputKeyClass(Text.class);
        job3.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job3, job2Out);
        FileOutputFormat.setOutputPath(job3, job3Out);
        if (!job3.waitForCompletion(true)) System.exit(1);

        // ---- Job4
        Configuration conf4 = new Configuration(conf);
        conf4.set("taskb.job1.output", job1Out.toString());

        Job job4 = Job.getInstance(conf4, "TaskB-Job4-EnrichCommonSupplierList");
        job4.setJarByClass(TaskB.class);
        job4.setMapperClass(EnrichMapper.class);
        job4.setReducerClass(EnrichReducer.class);
        job4.setOutputKeyClass(Text.class);
        job4.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job4, job3Out);
        FileOutputFormat.setOutputPath(job4, finalOut);

        System.exit(job4.waitForCompletion(true) ? 0 : 1);
    }
}
