package edu.cuhk.iems5730;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.*;
import org.apache.hadoop.mapreduce.lib.output.*;
import org.apache.hadoop.util.GenericOptionsParser;

import java.io.*;
import java.net.URI;
import java.util.*;

/**
 * Task E (Scalable for LARGE dataset):
 * For each company, find TOP-K (K=4) most similar companies (Jaccard over supplier sets)
 * and output common suppliers list for those pairs, formatted like Part (B):
 *   A:B, {C,E}, simscore
 *

 */
public class TaskE {

    // -------------------- Config --------------------
    private static final int TOP_K = 4;

    // Paths for intermediate outputs (suffixes)
    private static final String OUT_JOB1 = "_job1_counts";
    private static final String OUT_JOB2 = "_job2_pairs_by_supplier";
    private static final String OUT_JOB3 = "_job3_pair_common_counts";
    private static final String OUT_JOB4_TOPK = "_job4_topk";
    private static final String OUT_JOB4_PAIRS = "_job4_selected_pairs";
    private static final String OUT_JOB5_PAIR_SUP = "_job5_pair_suppliers_raw";
    private static final String OUT_JOB6_PAIR_SUP_AGG = "_job6_pair_suppliers_agg";

    // cache names
    private static final String CACHE_COUNTS = "company_counts.tsv";
    private static final String CACHE_SELECTED_PAIRS = "selected_pairs.tsv";

    // -------------------- Utilities --------------------

    /** Normalize pair (a,b) with a<b and encode into long key. */
    private static long pairKey(long a, long b) {
        long x = Math.min(a, b);
        long y = Math.max(a, b);
        return (x << 32) ^ (y & 0xffffffffL);
    }

    /** Decode pairKey to "a,b". */
    private static String pairKeyToString(long key) {
        long a = (key >>> 32);
        long b = (key & 0xffffffffL);
        // b might be negative if interpreted as signed int; fix:
        b = b & 0xffffffffL;
        return a + "," + b;
    }

    /** Parse "a,b" to pairKey. */
    private static long parsePairKey(String s) {
        String[] p = s.split(",");
        long a = Long.parseLong(p[0].trim());
        long b = Long.parseLong(p[1].trim());
        return pairKey(a, b);
    }

    /** Load a small TSV file from local cache into map: company->count */
    private static Map<Long, Integer> loadCountsFromLocalCache(File localFile) throws IOException {
        Map<Long, Integer> map = new HashMap<>(200_000);
        try (BufferedReader br = new BufferedReader(new FileReader(localFile))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                String[] parts = line.split("\\t");
                if (parts.length < 2) continue;
                long id = Long.parseLong(parts[0]);
                int c = Integer.parseInt(parts[1]);
                map.put(id, c);
            }
        }
        return map;
    }

    /** Load selected pair keys from local cache into a HashSet<Long>. */
    private static HashSet<Long> loadSelectedPairsFromLocalCache(File localFile) throws IOException {
        HashSet<Long> set = new HashSet<>(600_000);
        try (BufferedReader br = new BufferedReader(new FileReader(localFile))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                // accept "a,b" or "a\tb"
                if (line.contains("\t")) {
                    String[] parts = line.split("\\t");
                    if (parts.length >= 2) {
                        long a = Long.parseLong(parts[0]);
                        long b = Long.parseLong(parts[1]);
                        set.add(pairKey(a, b));
                    }
                } else {
                    set.add(parsePairKey(line));
                }
            }
        }
        return set;
    }

    // -------------------- Job1: company supplierCount --------------------

    public static class CountMapper extends Mapper<LongWritable, Text, LongWritable, LongWritable> {
        private final LongWritable outK = new LongWritable();
        private final LongWritable outV = new LongWritable();

        @Override
        public void map(LongWritable key, Text value, Context ctx) throws IOException, InterruptedException {
            String line = value.toString().trim();
            if (line.isEmpty()) return;
            String[] parts = line.split("\\s+");
            if (parts.length < 2) return;
            long buyer = Long.parseLong(parts[0]);
            long supplier = Long.parseLong(parts[1]);
            outK.set(buyer);
            outV.set(supplier);
            ctx.write(outK, outV);
        }
    }

    public static class CountReducer extends Reducer<LongWritable, LongWritable, LongWritable, IntWritable> {
        private final IntWritable outV = new IntWritable();

        @Override
        public void reduce(LongWritable buyer, Iterable<LongWritable> suppliers, Context ctx)
                throws IOException, InterruptedException {
            // dedup suppliers per buyer
            HashSet<Long> set = new HashSet<>();
            for (LongWritable s : suppliers) set.add(s.get());
            outV.set(set.size());
            ctx.write(buyer, outV);
        }
    }

    // -------------------- Job2: supplier -> buyers, emit pair -> 1 --------------------

    public static class SupplierToBuyerMapper extends Mapper<LongWritable, Text, LongWritable, LongWritable> {
        private final LongWritable outK = new LongWritable();
        private final LongWritable outV = new LongWritable();

        @Override
        public void map(LongWritable key, Text value, Context ctx) throws IOException, InterruptedException {
            String line = value.toString().trim();
            if (line.isEmpty()) return;
            String[] parts = line.split("\\s+");
            if (parts.length < 2) return;
            long buyer = Long.parseLong(parts[0]);
            long supplier = Long.parseLong(parts[1]);
            outK.set(supplier);
            outV.set(buyer);
            ctx.write(outK, outV);
        }
    }

    public static class GeneratePairsReducer extends Reducer<LongWritable, LongWritable, LongWritable, IntWritable> {
        private static final IntWritable ONE = new IntWritable(1);
        private final LongWritable outK = new LongWritable();

        @Override
        public void reduce(LongWritable supplier, Iterable<LongWritable> buyers, Context ctx)
                throws IOException, InterruptedException {
            // dedup buyers per supplier
            HashSet<Long> set = new HashSet<>();
            for (LongWritable b : buyers) set.add(b.get());
            if (set.size() < 2) return;

            long[] arr = new long[set.size()];
            int i = 0;
            for (Long b : set) arr[i++] = b;
            Arrays.sort(arr);

            // emit all pairs (a<b) contributed by this supplier
            for (int x = 0; x < arr.length; x++) {
                for (int y = x + 1; y < arr.length; y++) {
                    long pk = pairKey(arr[x], arr[y]);
                    outK.set(pk);
                    ctx.write(outK, ONE);
                }
            }
        }
    }

    // -------------------- Job3: sum commonCount per pair --------------------

    public static class PairCountMapper extends Mapper<LongWritable, Text, LongWritable, IntWritable> {
        private final LongWritable outK = new LongWritable();
        private final IntWritable outV = new IntWritable();

        @Override
        public void map(LongWritable key, Text value, Context ctx) throws IOException, InterruptedException {
            String line = value.toString().trim();
            if (line.isEmpty()) return;
            // input is "pairKey\t1"
            String[] parts = line.split("\\t");
            if (parts.length < 2) return;
            outK.set(Long.parseLong(parts[0]));
            outV.set(Integer.parseInt(parts[1]));
            ctx.write(outK, outV);
        }
    }

    public static class SumIntReducer extends Reducer<LongWritable, IntWritable, LongWritable, IntWritable> {
        private final IntWritable outV = new IntWritable();

        @Override
        public void reduce(LongWritable pairKey, Iterable<IntWritable> vals, Context ctx)
                throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable v : vals) sum += v.get();
            outV.set(sum);
            ctx.write(pairKey, outV);
        }
    }

    // -------------------- Job4: compute similarity + per-company TopK; output selected pairs --------------------

    public static class SimilarityTopKMapper extends Mapper<LongWritable, Text, LongWritable, Text> {
        private Map<Long, Integer> counts;
        private final LongWritable outK = new LongWritable();
        private final Text outV = new Text();

        @Override
        protected void setup(Context ctx) throws IOException {
            counts = new HashMap<>(200_000);

            URI[] cacheFiles = ctx.getCacheFiles();
            if (cacheFiles == null) throw new IOException("No cache file for company counts.");
            File local = null;
            for (URI u : cacheFiles) {
                if (u.getPath().endsWith(CACHE_COUNTS)) {
                    local = new File("./" + CACHE_COUNTS);
                    break;
                }
            }
            if (local == null || !local.exists()) {
                // fallback: first cache file
                local = new File("./" + new File(cacheFiles[0].getPath()).getName());
            }
            counts = loadCountsFromLocalCache(local);
        }

        @Override
        public void map(LongWritable key, Text value, Context ctx) throws IOException, InterruptedException {
            String line = value.toString().trim();
            if (line.isEmpty()) return;
            // input: "pairKey\tcommonCount"
            String[] parts = line.split("\\t");
            if (parts.length < 2) return;

            long pk = Long.parseLong(parts[0]);
            int common = Integer.parseInt(parts[1]);

            long a = (pk >>> 32);
            long b = (pk & 0xffffffffL) & 0xffffffffL;

            Integer ca = counts.get(a);
            Integer cb = counts.get(b);
            if (ca == null || cb == null) return;

            int denom = ca + cb - common;
            if (denom <= 0) return;
            double sim = (double) common / (double) denom;
            if (sim <= 0.0) return;

            // emit two directed records keyed by company
            // value: other,sim,pairKey
            outK.set(a);
            outV.set(b + "," + sim + "," + pk);
            ctx.write(outK, outV);

            outK.set(b);
            outV.set(a + "," + sim + "," + pk);
            ctx.write(outK, outV);
        }
    }

    private static class Cand {
        long other;
        double sim;
        long pairKey;

        Cand(long other, double sim, long pairKey) {
            this.other = other;
            this.sim = sim;
            this.pairKey = pairKey;
        }
    }

    /** Comparator for min-heap: keep worst on top (lowest sim, and for tie larger other) */
    private static final Comparator<Cand> WORST_FIRST = (c1, c2) -> {
        int cmp = Double.compare(c1.sim, c2.sim); // ascending sim
        if (cmp != 0) return cmp;
        // for equal sim, we prefer smaller other in final, so worst is larger other
        return Long.compare(c2.other, c1.other); // descending other => larger is "worse"
    };

    public static class TopKReducer extends Reducer<LongWritable, Text, Text, Text> {
        private MultipleOutputs<Text, Text> mos;

        @Override
        protected void setup(Context ctx) {
            mos = new MultipleOutputs<>(ctx);
        }

        @Override
        public void reduce(LongWritable company, Iterable<Text> vals, Context ctx)
                throws IOException, InterruptedException {

            PriorityQueue<Cand> pq = new PriorityQueue<>(TOP_K + 1, WORST_FIRST);

            for (Text t : vals) {
                String[] p = t.toString().split(",", 3);
                if (p.length < 3) continue;
                long other = Long.parseLong(p[0]);
                double sim = Double.parseDouble(p[1]);
                long pk = Long.parseLong(p[2]);

                Cand c = new Cand(other, sim, pk);
                pq.offer(c);
                if (pq.size() > TOP_K) pq.poll();
            }

            if (pq.isEmpty()) return;

            // sort final by best first: higher sim, tie smaller other
            ArrayList<Cand> list = new ArrayList<>(pq);
            list.sort((c1, c2) -> {
                int cmp = Double.compare(c2.sim, c1.sim); // descending sim
                if (cmp != 0) return cmp;
                return Long.compare(c1.other, c2.other);   // ascending other
            });

            long a = company.get();

            for (Cand c : list) {
                // topk output (directed)
                // company \t other,sim,pairKey
                mos.write("topk", new Text(Long.toString(a)),
                        new Text(c.other + "," + c.sim + "," + c.pairKey));

                // selected pairs output (normalized) as "a\tb"
                long pk = c.pairKey;
                long x = (pk >>> 32);
                long y = (pk & 0xffffffffL) & 0xffffffffL;
                mos.write("pairs", new Text(Long.toString(x)), new Text(Long.toString(y)));
            }
        }

        @Override
        protected void cleanup(Context ctx) throws IOException, InterruptedException {
            mos.close();
        }
    }

    // -------------------- Job5: for selected pairs only, compute (pair)->supplier occurrences --------------------

    public static class SupplierToBuyerForSelectedMapper extends Mapper<LongWritable, Text, LongWritable, LongWritable> {
        private final LongWritable outK = new LongWritable();
        private final LongWritable outV = new LongWritable();

        @Override
        public void map(LongWritable key, Text value, Context ctx) throws IOException, InterruptedException {
            String line = value.toString().trim();
            if (line.isEmpty()) return;
            String[] parts = line.split("\\s+");
            if (parts.length < 2) return;
            long buyer = Long.parseLong(parts[0]);
            long supplier = Long.parseLong(parts[1]);
            outK.set(supplier);
            outV.set(buyer);
            ctx.write(outK, outV);
        }
    }

    public static class SelectedPairsSupplierReducer extends Reducer<LongWritable, LongWritable, Text, LongWritable> {
        private HashSet<Long> selectedPairs;
        private final Text outK = new Text();
        private final LongWritable outV = new LongWritable();

        @Override
        protected void setup(Context ctx) throws IOException {
            URI[] cacheFiles = ctx.getCacheFiles();
            if (cacheFiles == null) throw new IOException("No cache file for selected pairs.");
            File local = null;
            for (URI u : cacheFiles) {
                if (u.getPath().endsWith(CACHE_SELECTED_PAIRS)) {
                    local = new File("./" + CACHE_SELECTED_PAIRS);
                    break;
                }
            }
            if (local == null || !local.exists()) {
                local = new File("./" + new File(cacheFiles[0].getPath()).getName());
            }
            selectedPairs = loadSelectedPairsFromLocalCache(local);
        }

        @Override
        public void reduce(LongWritable supplier, Iterable<LongWritable> buyers, Context ctx)
                throws IOException, InterruptedException {

            HashSet<Long> set = new HashSet<>();
            for (LongWritable b : buyers) set.add(b.get());
            if (set.size() < 2) return;

            long[] arr = new long[set.size()];
            int i = 0;
            for (Long b : set) arr[i++] = b;
            Arrays.sort(arr);

            long sup = supplier.get();
            outV.set(sup);

            // enumerate buyer pairs, only emit if in selectedPairs
            for (int x = 0; x < arr.length; x++) {
                for (int y = x + 1; y < arr.length; y++) {
                    long pk = pairKey(arr[x], arr[y]);
                    if (selectedPairs.contains(pk)) {
                        outK.set(pairKeyToString(pk)); // "a,b"
                        ctx.write(outK, outV);         // pair -> supplier
                    }
                }
            }
        }
    }

    // -------------------- Job6: aggregate suppliers per pair --------------------

    public static class PairSupplierMapper extends Mapper<LongWritable, Text, Text, LongWritable> {
        private final Text outK = new Text();
        private final LongWritable outV = new LongWritable();

        @Override
        public void map(LongWritable key, Text value, Context ctx) throws IOException, InterruptedException {
            String line = value.toString().trim();
            if (line.isEmpty()) return;
            // "a,b \t supplier"
            String[] parts = line.split("\\t");
            if (parts.length < 2) return;
            outK.set(parts[0]);
            outV.set(Long.parseLong(parts[1]));
            ctx.write(outK, outV);
        }
    }

    public static class PairSupplierAggReducer extends Reducer<Text, LongWritable, Text, Text> {
        private final Text outV = new Text();

        @Override
        public void reduce(Text pair, Iterable<LongWritable> suppliers, Context ctx)
                throws IOException, InterruptedException {

            TreeSet<Long> set = new TreeSet<>();
            for (LongWritable s : suppliers) set.add(s.get());
            if (set.isEmpty()) return;

            StringBuilder sb = new StringBuilder();
            sb.append("{");
            boolean first = true;
            for (Long s : set) {
                if (!first) sb.append(",");
                sb.append(s);
                first = false;
            }
            sb.append("}");
            outV.set(sb.toString());
            ctx.write(pair, outV);
        }
    }

    // -------------------- Job7: join TopK with supplier lists and output final lines --------------------

    public static class TopKJoinMapper extends Mapper<LongWritable, Text, Text, Text> {
        private final Text outK = new Text();
        private final Text outV = new Text();

        @Override
        public void map(LongWritable key, Text value, Context ctx) throws IOException, InterruptedException {
            // topk output: company \t other,sim,pairKey
            String line = value.toString().trim();
            if (line.isEmpty()) return;
            String[] parts = line.split("\\t");
            if (parts.length < 2) return;

            long company = Long.parseLong(parts[0]);
            String[] p2 = parts[1].split(",", 3);
            if (p2.length < 3) return;
            long other = Long.parseLong(p2[0]);
            double sim = Double.parseDouble(p2[1]);
            long pk = Long.parseLong(p2[2]);

            String pairStr = pairKeyToString(pk); // normalized "a,b"
            outK.set(pairStr);
            // store directed top record
            outV.set("TOP:" + company + ":" + other + ":" + sim);
            ctx.write(outK, outV);
        }
    }

    public static class PairSupJoinMapper extends Mapper<LongWritable, Text, Text, Text> {
        private final Text outK = new Text();
        private final Text outV = new Text();

        @Override
        public void map(LongWritable key, Text value, Context ctx) throws IOException, InterruptedException {
            // pairSupAgg: "a,b \t {s1,s2}"
            String line = value.toString().trim();
            if (line.isEmpty()) return;
            String[] parts = line.split("\\t");
            if (parts.length < 2) return;
            outK.set(parts[0]);
            outV.set("SUP:" + parts[1]);
            ctx.write(outK, outV);
        }
    }

    public static class FinalJoinReducer extends Reducer<Text, Text, Text, NullWritable> {
        @Override
        public void reduce(Text pair, Iterable<Text> vals, Context ctx)
                throws IOException, InterruptedException {
            String supList = "{}";
            List<String> tops = new ArrayList<>(4);

            for (Text t : vals) {
                String s = t.toString();
                if (s.startsWith("SUP:")) supList = s.substring(4);
                else if (s.startsWith("TOP:")) tops.add(s.substring(4));
            }

            if (tops.isEmpty()) return;

            for (String rec : tops) {
                // rec: company:other:sim
                String[] p = rec.split(":");
                if (p.length < 3) continue;
                String company = p[0];
                String other = p[1];
                String sim = p[2];

                // format: A:B, {C,E}, simscore
                String outLine = company + ":" + other + ", " + supList + ", " + sim;
                ctx.write(new Text(outLine), NullWritable.get());
            }
        }
    }

    // -------------------- Driver --------------------

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();

        if (otherArgs.length != 2) {
            System.err.println("Usage: TaskE <large_relation_input> <output_path>");
            System.exit(2);
        }

        String input = otherArgs[0];
        String outBase = otherArgs[1];

        Path job1Out = new Path(outBase + OUT_JOB1);
        Path job2Out = new Path(outBase + OUT_JOB2);
        Path job3Out = new Path(outBase + OUT_JOB3);
        Path job4TopOut = new Path(outBase + OUT_JOB4_TOPK);
        Path job4PairsOut = new Path(outBase + OUT_JOB4_PAIRS);
        Path job5Out = new Path(outBase + OUT_JOB5_PAIR_SUP);
        Path job6Out = new Path(outBase + OUT_JOB6_PAIR_SUP_AGG);
        Path finalOut = new Path(outBase);

        // clean outputs if exist
        FileSystem fs = FileSystem.get(conf);
        fs.delete(job1Out, true);
        fs.delete(job2Out, true);
        fs.delete(job3Out, true);
        fs.delete(job4TopOut, true);
        fs.delete(job4PairsOut, true);
        fs.delete(job5Out, true);
        fs.delete(job6Out, true);
        fs.delete(finalOut, true);

        // ---------------- Job1: company supplierCount ----------------
        Job job1 = Job.getInstance(conf, "TaskE-Job1-CompanySupplierCount");
        job1.setJarByClass(TaskE.class);
        job1.setMapperClass(CountMapper.class);
        job1.setReducerClass(CountReducer.class);
        job1.setMapOutputKeyClass(LongWritable.class);
        job1.setMapOutputValueClass(LongWritable.class);
        job1.setOutputKeyClass(LongWritable.class);
        job1.setOutputValueClass(IntWritable.class);
        job1.setNumReduceTasks(16);
        TextInputFormat.addInputPath(job1, new Path(input));
        FileOutputFormat.setOutputPath(job1, job1Out);
        if (!job1.waitForCompletion(true)) System.exit(1);

        // ---------------- Job2: supplier->buyers -> pair->1 ----------------
        Job job2 = Job.getInstance(conf, "TaskE-Job2-GeneratePairsBySupplier");
        job2.setJarByClass(TaskE.class);
        job2.setMapperClass(SupplierToBuyerMapper.class);
        job2.setReducerClass(GeneratePairsReducer.class);
        job2.setMapOutputKeyClass(LongWritable.class);
        job2.setMapOutputValueClass(LongWritable.class);
        job2.setOutputKeyClass(LongWritable.class);
        job2.setOutputValueClass(IntWritable.class);
        job2.setNumReduceTasks(32);
        TextInputFormat.addInputPath(job2, new Path(input));
        FileOutputFormat.setOutputPath(job2, job2Out);
        if (!job2.waitForCompletion(true)) System.exit(1);

        // ---------------- Job3: sum common counts ----------------
        Job job3 = Job.getInstance(conf, "TaskE-Job3-SumCommonCounts");
        job3.setJarByClass(TaskE.class);
        job3.setMapperClass(PairCountMapper.class);
        job3.setCombinerClass(SumIntReducer.class);
        job3.setReducerClass(SumIntReducer.class);
        job3.setMapOutputKeyClass(LongWritable.class);
        job3.setMapOutputValueClass(IntWritable.class);
        job3.setOutputKeyClass(LongWritable.class);
        job3.setOutputValueClass(IntWritable.class);
        job3.setNumReduceTasks(32);
        TextInputFormat.addInputPath(job3, job2Out);
        FileOutputFormat.setOutputPath(job3, job3Out);
        if (!job3.waitForCompletion(true)) System.exit(1);

        // ---------------- Job4: similarity + TopK; output selected pairs ----------------
        Job job4 = Job.getInstance(conf, "TaskE-Job4-TopKSimilarity");
        job4.setJarByClass(TaskE.class);

        // add cache file (company counts) - rename to fixed local name
        // use the first part file(s); easiest: add the whole directory via glob is not supported,
        // so we copy merged counts to a single file.
        Path mergedCounts = new Path(outBase + "_cache_company_counts.tsv");
        fs.delete(mergedCounts, true);
        mergeToSingleFile(fs, job1Out, mergedCounts);

        job4.addCacheFile(new URI(mergedCounts.toString() + "#" + CACHE_COUNTS));

        job4.setMapperClass(SimilarityTopKMapper.class);
        job4.setReducerClass(TopKReducer.class);
        job4.setMapOutputKeyClass(LongWritable.class);
        job4.setMapOutputValueClass(Text.class);
        job4.setOutputKeyClass(Text.class);
        job4.setOutputValueClass(Text.class);
        job4.setNumReduceTasks(32);

        MultipleOutputs.addNamedOutput(job4, "topk", TextOutputFormat.class, Text.class, Text.class);
        MultipleOutputs.addNamedOutput(job4, "pairs", TextOutputFormat.class, Text.class, Text.class);

        TextInputFormat.addInputPath(job4, job3Out);
        FileOutputFormat.setOutputPath(job4, new Path(outBase + "_job4_tmp"));
        if (!job4.waitForCompletion(true)) System.exit(1);

        // Move MultipleOutputs results to stable paths
        moveNamedOutput(fs, new Path(outBase + "_job4_tmp"), "topk", job4TopOut);
        moveNamedOutput(fs, new Path(outBase + "_job4_tmp"), "pairs", job4PairsOut);
        fs.delete(new Path(outBase + "_job4_tmp"), true);

        // ---------------- Job5: compute pair->supplier only for selected pairs ----------------
        Job job5 = Job.getInstance(conf, "TaskE-Job5-SelectedPairsCommonSuppliers");
        job5.setJarByClass(TaskE.class);

        Path mergedPairs = new Path(outBase + "_cache_selected_pairs.tsv");
        fs.delete(mergedPairs, true);
        mergeToSingleFile(fs, job4PairsOut, mergedPairs);

        job5.addCacheFile(new URI(mergedPairs.toString() + "#" + CACHE_SELECTED_PAIRS));

        job5.setMapperClass(SupplierToBuyerForSelectedMapper.class);
        job5.setReducerClass(SelectedPairsSupplierReducer.class);
        job5.setMapOutputKeyClass(LongWritable.class);
        job5.setMapOutputValueClass(LongWritable.class);
        job5.setOutputKeyClass(Text.class);
        job5.setOutputValueClass(LongWritable.class);
        job5.setNumReduceTasks(32);

        TextInputFormat.addInputPath(job5, new Path(input));
        FileOutputFormat.setOutputPath(job5, job5Out);
        if (!job5.waitForCompletion(true)) System.exit(1);

        // ---------------- Job6: aggregate suppliers list per pair ----------------
        Job job6 = Job.getInstance(conf, "TaskE-Job6-AggregatePairSuppliers");
        job6.setJarByClass(TaskE.class);
        job6.setMapperClass(PairSupplierMapper.class);
        job6.setReducerClass(PairSupplierAggReducer.class);
        job6.setMapOutputKeyClass(Text.class);
        job6.setMapOutputValueClass(LongWritable.class);
        job6.setOutputKeyClass(Text.class);
        job6.setOutputValueClass(Text.class);
        job6.setNumReduceTasks(16);

        TextInputFormat.addInputPath(job6, job5Out);
        FileOutputFormat.setOutputPath(job6, job6Out);
        if (!job6.waitForCompletion(true)) System.exit(1);

        // ---------------- Job7: join topk with supplier lists and output final lines ----------------
        Job job7 = Job.getInstance(conf, "TaskE-Job7-FinalJoinAndFormat");
        job7.setJarByClass(TaskE.class);

        MultipleInputs.addInputPath(job7, job4TopOut, TextInputFormat.class, TopKJoinMapper.class);
        MultipleInputs.addInputPath(job7, job6Out, TextInputFormat.class, PairSupJoinMapper.class);

        job7.setReducerClass(FinalJoinReducer.class);
        job7.setMapOutputKeyClass(Text.class);
        job7.setMapOutputValueClass(Text.class);
        job7.setOutputKeyClass(Text.class);
        job7.setOutputValueClass(NullWritable.class);
        job7.setNumReduceTasks(16);

        FileOutputFormat.setOutputPath(job7, finalOut);
        System.exit(job7.waitForCompletion(true) ? 0 : 1);
    }

    // -------------------- Helper: merge part files to single cache file --------------------

    private static void mergeToSingleFile(FileSystem fs, Path dir, Path outFile) throws IOException {
        // concatenate all part-* into one file in HDFS (simple and OK for ~100k-400k lines)
        FSDataOutputStream out = fs.create(outFile, true);
        try {
            FileStatus[] files = fs.listStatus(dir, path -> path.getName().startsWith("part-"));
            Arrays.sort(files, Comparator.comparing(f -> f.getPath().getName()));
            byte[] buf = new byte[1 << 20];
            for (FileStatus f : files) {
                try (FSDataInputStream in = fs.open(f.getPath())) {
                    int r;
                    while ((r = in.read(buf)) > 0) out.write(buf, 0, r);
                }
            }
        } finally {
            out.close();
        }
    }

    private static void moveNamedOutput(FileSystem fs, Path tmpDir, String named, Path targetDir) throws IOException {
        fs.delete(targetDir, true);
        fs.mkdirs(targetDir);

        // named outputs are usually like: <tmpDir>/<named>-m-00000 or <named>-r-00000
        FileStatus[] files = fs.listStatus(tmpDir, path -> path.getName().startsWith(named + "-"));
        for (FileStatus f : files) {
            Path dst = new Path(targetDir, "part-" + f.getPath().getName()); // keep distinct
            fs.rename(f.getPath(), dst);
        }
    }
}

