package edu.cuhk.iems5730;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.*;
import java.net.URI;
import java.util.*;

/**
 * Task B:
 * For each company, find TOP K (K=3) most similar companies and their common suppliers.
 *
 * Similarity = |Common Suppliers| / |Union Suppliers|
 *
 * Recommended scalable design (4 MR jobs):
 *  Job1: company -> size|sorted_unique_supplier_list
 *  Job2: supplier -> list of companies ; emit pair -> supplier (each common supplier contributes one record)
 *  Job3: pair -> aggregate unique common suppliers ; emit pair -> commonCount \t {sup1,sup2,...}
 *  Job4: (load company sizes from Job1 outputs via DistributedCache)
 *        pair -> similarity ; emit company -> (other, commonSuppliers, similarity) ; reduce -> TopK
 */
public class TaskB {

    // -----------------------------
    // Job1: Build Supplier List
    // output: company \t size|sup1,sup2,...
    // -----------------------------
    public static class Job1Mapper extends Mapper<LongWritable, Text, Text, Text> {
        private final Text k = new Text();
        private final Text v = new Text();

        @Override
        public void map(LongWritable key, Text value, Context ctx) throws IOException, InterruptedException {
            String line = value.toString().trim();
            if (line.isEmpty()) return;
            String[] parts = line.split("\\s+");
            if (parts.length < 2) return;

            k.set(parts[0]);     // buyer/company
            v.set(parts[1]);     // supplier
            ctx.write(k, v);
        }
    }

    public static class Job1Reducer extends Reducer<Text, Text, Text, Text> {
        private final Text outV = new Text();

        @Override
        public void reduce(Text company, Iterable<Text> suppliers, Context ctx) throws IOException, InterruptedException {
            TreeSet<String> set = new TreeSet<>();
            for (Text s : suppliers) set.add(s.toString());
            if (set.isEmpty()) return;

            int size = set.size();
            String joined = String.join(",", set);
            outV.set(size + "|" + joined);
            ctx.write(company, outV);
        }
    }

    // -----------------------------
    // Job2: Inverted index by supplier
    // input: company \t size|sup1,sup2,...
    // map: emit supplier -> company|size
    // reduce: for each supplier, generate all pairs; emit pair -> supplier
    // -----------------------------
    public static class Job2Mapper extends Mapper<LongWritable, Text, Text, Text> {
        private final Text outK = new Text();
        private final Text outV = new Text();

        @Override
        public void map(LongWritable key, Text value, Context ctx) throws IOException, InterruptedException {
            String line = value.toString().trim();
            if (line.isEmpty()) return;

            String[] kv = line.split("\\t");
            if (kv.length < 2) return;

            String company = kv[0];
            String payload = kv[1]; // size|sup1,sup2,...
            String[] sp = payload.split("\\|", 2);
            if (sp.length < 2) return;

            String sizeStr = sp[0];
            String supList = sp[1];
            if (supList.isEmpty()) return;

            String[] sups = supList.split(",");
            for (String sup : sups) {
                sup = sup.trim();
                if (sup.isEmpty()) continue;
                outK.set(sup);
                outV.set(company + "|" + sizeStr);
                ctx.write(outK, outV);
            }
        }
    }

    public static class Job2Reducer extends Reducer<Text, Text, Text, Text> {
        private final Text outK = new Text();
        private final Text outV = new Text();

        @Override
        public void reduce(Text supplier, Iterable<Text> values, Context ctx) throws IOException, InterruptedException {
            // collect companies that have this supplier
            ArrayList<String> companies = new ArrayList<>();
            for (Text t : values) companies.add(t.toString());

            int n = companies.size();
            if (n < 2) return;

            // pair all companies under same supplier
            for (int i = 0; i < n; i++) {
                String ci = companies.get(i).split("\\|", 2)[0];
                for (int j = i + 1; j < n; j++) {
                    String cj = companies.get(j).split("\\|", 2)[0];

                    String a = ci, b = cj;
                    if (a.compareTo(b) > 0) { String tmp = a; a = b; b = tmp; }
                    outK.set(a + "," + b);
                    outV.set(supplier.toString()); // one common supplier record
                    ctx.write(outK, outV);
                }
            }
        }
    }

    // -----------------------------
    // Job3: Aggregate pair -> unique common suppliers
    // input: pair \t supplier
    // output: pair \t commonCount \t {sup1,sup2,...}
    // -----------------------------
    public static class Job3Mapper extends Mapper<LongWritable, Text, Text, Text> {
        private final Text outK = new Text();
        private final Text outV = new Text();

        @Override
        public void map(LongWritable key, Text value, Context ctx) throws IOException, InterruptedException {
            String line = value.toString().trim();
            if (line.isEmpty()) return;
            String[] kv = line.split("\\t");
            if (kv.length < 2) return;

            outK.set(kv[0]);   // pair
            outV.set(kv[1]);   // supplier
            ctx.write(outK, outV);
        }
    }

    public static class Job3Reducer extends Reducer<Text, Text, Text, Text> {
        private final Text outV = new Text();

        @Override
        public void reduce(Text pair, Iterable<Text> suppliers, Context ctx) throws IOException, InterruptedException {
            TreeSet<String> set = new TreeSet<>();
            for (Text s : suppliers) set.add(s.toString());
            if (set.isEmpty()) return;

            int common = set.size();
            String commonList = "{" + String.join(",", set) + "}";
            outV.set(common + "\t" + commonList);
            ctx.write(pair, outV);
        }
    }

    // -----------------------------
    // Job4: Compute similarity + TopK per company
    // input: pair \t commonCount \t {sup...}
    // load company sizes from Job1 outputs via DistributedCache
    // output format (same style as your TaskA/old TaskB):
    //   company:other, {commonSuppliers}, similarity
    // -----------------------------
    public static class Job4Mapper extends Mapper<LongWritable, Text, Text, Text> {
        private final Map<String, Integer> companySize = new HashMap<>();
        private final Text outK = new Text();
        private final Text outV = new Text();

        @Override
        protected void setup(Context ctx) throws IOException {
            URI[] files = ctx.getCacheFiles();
            if (files == null) return;

            for (URI uri : files) {
                Path p = new Path(uri.getPath());
                FileSystem fs = FileSystem.get(ctx.getConfiguration());
                try (BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(p)))) {
                    String line;
                    while ((line = br.readLine()) != null) {
                        line = line.trim();
                        if (line.isEmpty()) continue;
                        String[] kv = line.split("\\t");
                        if (kv.length < 2) continue;

                        String company = kv[0];
                        String[] sp = kv[1].split("\\|", 2);
                        if (sp.length < 1) continue;
                        try {
                            int sz = Integer.parseInt(sp[0]);
                            companySize.put(company, sz);
                        } catch (NumberFormatException ignored) {}
                    }
                }
            }
        }

        @Override
        public void map(LongWritable key, Text value, Context ctx) throws IOException, InterruptedException {
            String line = value.toString().trim();
            if (line.isEmpty()) return;

            // pair \t commonCount \t {sup...}
            String[] kv = line.split("\\t");
            if (kv.length < 3) return;

            String pair = kv[0];
            int commonCount;
            try {
                commonCount = Integer.parseInt(kv[1]);
            } catch (NumberFormatException e) {
                return;
            }
            String commonSuppliers = kv[2];

            String[] ab = pair.split(",", 2);
            if (ab.length < 2) return;
            String a = ab[0], b = ab[1];

            Integer sa = companySize.get(a);
            Integer sb = companySize.get(b);
            if (sa == null || sb == null) return;

            int union = sa + sb - commonCount;
            if (union <= 0) return;

            double sim = (double) commonCount / (double) union;
            if (sim <= 0) return;

            String simStr = String.format("%.6f", sim);

            // emit for a
            outK.set(a);
            outV.set(b + ", " + commonSuppliers + ", " + simStr);
            ctx.write(outK, outV);

            // emit for b
            outK.set(b);
            outV.set(a + ", " + commonSuppliers + ", " + simStr);
            ctx.write(outK, outV);
        }
    }

    public static class Job4TopKReducer extends Reducer<Text, Text, Text, Text> {
        private int K;
        private final Text outKey = new Text();

        private static class Rec {
            String other;
            String common;
            double sim;
            Rec(String o, String c, double s) { other=o; common=c; sim=s; }
        }

        @Override
        protected void setup(Context ctx) {
            K = ctx.getConfiguration().getInt("topk.k", 3);
        }

        @Override
        public void reduce(Text company, Iterable<Text> values, Context ctx) throws IOException, InterruptedException {
            PriorityQueue<Rec> pq = new PriorityQueue<>(K + 1, (x, y) -> {
                int c = Double.compare(x.sim, y.sim); // min-heap by similarity
                if (c != 0) return c;
                return y.other.compareTo(x.other);    // tie-break
            });

            for (Text t : values) {
                String s = t.toString();
                // other, {common}, sim
                // split into 3 parts, but common contains braces and commas, so split carefully:
                // We stored: other + ", " + commonSuppliers + ", " + simStr
                int firstComma = s.indexOf(", ");
                if (firstComma < 0) continue;
                String other = s.substring(0, firstComma);

                int lastComma = s.lastIndexOf(", ");
                if (lastComma <= firstComma) continue;
                String common = s.substring(firstComma + 2, lastComma);
                String simStr = s.substring(lastComma + 2);

                double sim;
                try { sim = Double.parseDouble(simStr); } catch (NumberFormatException e) { continue; }

                pq.offer(new Rec(other, common, sim));
                if (pq.size() > K) pq.poll();
            }

            ArrayList<Rec> res = new ArrayList<>(pq);
            res.sort((x, y) -> {
                int c = Double.compare(y.sim, x.sim);
                if (c != 0) return c;
                return x.other.compareTo(y.other);
            });

            for (Rec r : res) {
                outKey.set(company.toString() + ":" + r.other + ", " + r.common + ", " + String.format("%.6f", r.sim));
                ctx.write(outKey, new Text(""));
            }
        }
    }

    // -----------------------------
    // Driver
    // -----------------------------
    public static void main(String[] args) throws Exception {
        if (args.length < 2 || args.length > 3) {
            System.err.println("Usage: TaskB <input_path> <output_path> [K]");
            System.exit(1);
        }

        String input = args[0];
        String output = args[1];
        int K = (args.length == 3) ? Integer.parseInt(args[2]) : 3;

        Configuration conf = new Configuration();
        conf.setInt("topk.k", K);

        FileSystem fs = FileSystem.get(conf);
        Path outFinal = new Path(output);
        Path out1 = new Path(output + "_job1");
        Path out2 = new Path(output + "_job2");
        Path out3 = new Path(output + "_job3");

        // clean old outputs
        fs.delete(outFinal, true);
        fs.delete(out1, true);
        fs.delete(out2, true);
        fs.delete(out3, true);

        // ---- Job1
        Job job1 = Job.getInstance(conf, "TaskB-Job1-BuildSupplierLists");
        job1.setJarByClass(TaskB.class);
        job1.setMapperClass(Job1Mapper.class);
        job1.setReducerClass(Job1Reducer.class);
        job1.setMapOutputKeyClass(Text.class);
        job1.setMapOutputValueClass(Text.class);
        job1.setOutputKeyClass(Text.class);
        job1.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(job1, new Path(input));
        FileOutputFormat.setOutputPath(job1, out1);

        if (!job1.waitForCompletion(true)) System.exit(1);

        // ---- Job2
        Job job2 = Job.getInstance(conf, "TaskB-Job2-InvertedIndexPairs");
        job2.setJarByClass(TaskB.class);
        job2.setMapperClass(Job2Mapper.class);
        job2.setReducerClass(Job2Reducer.class);
        job2.setMapOutputKeyClass(Text.class);
        job2.setMapOutputValueClass(Text.class);
        job2.setOutputKeyClass(Text.class);
        job2.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(job2, out1);
        FileOutputFormat.setOutputPath(job2, out2);

        if (!job2.waitForCompletion(true)) System.exit(1);

        // ---- Job3
        Job job3 = Job.getInstance(conf, "TaskB-Job3-AggregateCommonSuppliers");
        job3.setJarByClass(TaskB.class);
        job3.setMapperClass(Job3Mapper.class);
        job3.setReducerClass(Job3Reducer.class);
        job3.setMapOutputKeyClass(Text.class);
        job3.setMapOutputValueClass(Text.class);
        job3.setOutputKeyClass(Text.class);
        job3.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(job3, out2);
        FileOutputFormat.setOutputPath(job3, out3);

        if (!job3.waitForCompletion(true)) System.exit(1);

        // ---- Job4
        Job job4 = Job.getInstance(conf, "TaskB-Job4-TopKPerCompany");
        job4.setJarByClass(TaskB.class);
        job4.setMapperClass(Job4Mapper.class);
        job4.setReducerClass(Job4TopKReducer.class);
        job4.setMapOutputKeyClass(Text.class);
        job4.setMapOutputValueClass(Text.class);
        job4.setOutputKeyClass(Text.class);
        job4.setOutputValueClass(Text.class);

        // Add Job1 output part files to DistributedCache
        FileStatus[] partFiles = fs.listStatus(out1, path -> path.getName().startsWith("part-"));
        if (partFiles != null) {
            for (FileStatus st : partFiles) {
                job4.addCacheFile(st.getPath().toUri());
            }
        }

        FileInputFormat.addInputPath(job4, out3);
        FileOutputFormat.setOutputPath(job4, outFinal);

        System.exit(job4.waitForCompletion(true) ? 0 : 1);
    }
}
