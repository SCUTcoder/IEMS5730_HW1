package edu.cuhk.iems5730;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import java.io.*;
import java.net.URI;
import java.util.*;

/**
 * HW1 Task E:
 * For each company, find TOP-K (K=4) most similar companies in LARGE dataset,
 * and output common suppliers list + similarity score in the format of Task B:
 *   A:B, {C,E}, simscore
 *
 * Similarity(A,B) = |Sup(A) ∩ Sup(B)| / |Sup(A) ∪ Sup(B)|
 * If tie in similarity: pick smaller company IDs (numeric meaning) first.
 *
 * IMPORTANT: Company IDs can exceed Long range => treat IDs as String/Text.
 *
 * Pipeline (4 jobs):
 *  Job1: Dedup suppliers per buyer; output:
 *        (a) supplier -> buyer edges (distinct)  [for inverted index]
 *        (b) buyer -> degree (#distinct suppliers) [for similarity denominator]
 *  Job2: Build supplier -> buyers list; emit buyer-pairs with common supplier.
 *  Job3: Aggregate by pair -> (commonCount, commonSuppliersList)
 *  Job4: Join degrees, compute similarity, emit to each side, reducer selects TopK=4.
 */
public class TaskE {

    // -------------------- Helpers --------------------

    /** numeric compare for non-negative integer strings without parsing */
    static int compareNumericId(String a, String b) {
        // strip leading zeros for stable compare (optional, safe)
        a = stripLeadingZeros(a);
        b = stripLeadingZeros(b);
        if (a.length() != b.length()) return Integer.compare(a.length(), b.length());
        return a.compareTo(b);
    }

    static String stripLeadingZeros(String s) {
        int i = 0;
        while (i < s.length() - 1 && s.charAt(i) == '0') i++;
        return s.substring(i);
    }

    static boolean isAllDigits(String s) {
        if (s == null || s.isEmpty()) return false;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c < '0' || c > '9') return false;
        }
        return true;
    }

    /** canonical pair key: smaller id first (numeric meaning) */
    static String canonPair(String a, String b) {
        if (compareNumericId(a, b) <= 0) return a + "," + b;
        return b + "," + a;
    }

    // -------------------- Job1: buyer -> distinct suppliers; outputs:
    //  - "sb" named output: supplier \t buyer   (for Job2)
    //  - "deg" named output: buyer \t degree    (for Job4)
    // --------------------

    public static class Job1Mapper extends Mapper<LongWritable, Text, Text, Text> {
        private final Text outKey = new Text();
        private final Text outVal = new Text();

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString().trim();
            if (line.isEmpty()) return;

            String[] parts = line.split("\\s+");
            if (parts.length < 2) return;

            String buyer = parts[0].trim();
            String supplier = parts[1].trim();

            // Robust guards (keep; large data sometimes has dirty lines)
            if (!isAllDigits(buyer) || !isAllDigits(supplier)) return;

            outKey.set(buyer);
            outVal.set(supplier);
            context.write(outKey, outVal);
        }
    }

    public static class Job1Reducer extends Reducer<Text, Text, Text, Text> {
        private MultipleOutputs<Text, Text> mos;

        @Override
        protected void setup(Context context) {
            mos = new MultipleOutputs<>(context);
        }

        @Override
        protected void reduce(Text buyer, Iterable<Text> suppliers, Context context)
                throws IOException, InterruptedException {

            // Dedup suppliers of this buyer
            HashSet<String> set = new HashSet<>();
            for (Text s : suppliers) {
                String sup = s.toString();
                if (!sup.isEmpty()) set.add(sup);
            }

            // (b) output degree
            mos.write("deg", buyer, new Text(String.valueOf(set.size())));

            // (a) output supplier->buyer edges (distinct)
            // format: supplier \t buyer
            for (String sup : set) {
                mos.write("sb", new Text(sup), buyer);
            }
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            mos.close();
        }
    }

    // -------------------- Job2: supplier -> buyers; emit buyer-pairs with common supplier
    // Input: supplier \t buyer  (from Job1 "sb")
    // Output: pairKey("a,b") \t supplier
    // --------------------

    public static class Job2Mapper extends Mapper<LongWritable, Text, Text, Text> {
        private final Text outKey = new Text();
        private final Text outVal = new Text();

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString().trim();
            if (line.isEmpty()) return;
            String[] parts = line.split("\\s+");
            if (parts.length < 2) return;

            String supplier = parts[0].trim();
            String buyer = parts[1].trim();
            if (!isAllDigits(supplier) || !isAllDigits(buyer)) return;

            outKey.set(supplier);
            outVal.set(buyer);
            context.write(outKey, outVal);
        }
    }

    public static class Job2Reducer extends Reducer<Text, Text, Text, Text> {
        private final Text outKey = new Text();
        private final Text outVal = new Text();

        @Override
        protected void reduce(Text supplier, Iterable<Text> buyers, Context context)
                throws IOException, InterruptedException {

            // Dedup buyers for this supplier (defensive)
            ArrayList<String> list = new ArrayList<>();
            HashSet<String> seen = new HashSet<>();
            for (Text b : buyers) {
                String s = b.toString();
                if (s.isEmpty()) continue;
                if (seen.add(s)) list.add(s);
            }

            int n = list.size();
            if (n < 2) return;

            // Generate all unordered pairs (a,b) who share this supplier
            // NOTE: This can be heavy for very high-degree suppliers, but it's the standard inverted-index approach.
            for (int i = 0; i < n; i++) {
                String a = list.get(i);
                for (int j = i + 1; j < n; j++) {
                    String b = list.get(j);
                    outKey.set(canonPair(a, b));
                    outVal.set(supplier.toString());
                    context.write(outKey, outVal);
                }
            }
        }
    }

    // -------------------- Job3: aggregate by pair -> common suppliers list + count
    // Input: pairKey("a,b") \t supplier
    // Output: pairKey \t commonCount \t supplier1,supplier2,...
    // --------------------

    public static class Job3Mapper extends Mapper<LongWritable, Text, Text, Text> {
        private final Text outKey = new Text();
        private final Text outVal = new Text();

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString().trim();
            if (line.isEmpty()) return;
            String[] parts = line.split("\\s+");
            if (parts.length < 2) return;

            String pair = parts[0].trim();
            String supplier = parts[1].trim();
            if (pair.isEmpty() || supplier.isEmpty()) return;

            outKey.set(pair);
            outVal.set(supplier);
            context.write(outKey, outVal);
        }
    }

    public static class Job3Reducer extends Reducer<Text, Text, Text, Text> {
        private final Text outVal = new Text();

        @Override
        protected void reduce(Text pair, Iterable<Text> suppliers, Context context)
                throws IOException, InterruptedException {

            // Dedup suppliers for this pair (they come from different suppliers reducers, but safe)
            // Keep as list to output "{...}"
            HashSet<String> set = new HashSet<>();
            for (Text s : suppliers) {
                String sup = s.toString();
                if (!sup.isEmpty()) set.add(sup);
            }

            if (set.isEmpty()) return;

            // For readability: sort suppliers by numeric id asc (optional)
            ArrayList<String> list = new ArrayList<>(set);
            list.sort(TaskE::compareNumericId);

            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < list.size(); i++) {
                if (i > 0) sb.append(",");
                sb.append(list.get(i));
            }

            int common = list.size();
            outVal.set(common + "\t" + sb);
            context.write(pair, outVal);
        }
    }

    // -------------------- Job4: compute similarity + TopK for each company
    // Input: pairKey \t commonCount \t supplierCSV
    // Side input: degrees file (buyer \t deg) from Job1 "deg"
    // Output lines: A:B, {sup1,sup2}, simscore  (Top 4 per A)
    // --------------------

    public static class Job4Mapper extends Mapper<LongWritable, Text, Text, Text> {

        private final Text outKey = new Text();
        private final Text outVal = new Text();
        private final HashMap<String, Integer> degMap = new HashMap<>();

        @Override
        protected void setup(Context context) throws IOException {
            // Load degree side file(s) from DistributedCache
            URI[] cacheFiles = context.getCacheFiles();
            if (cacheFiles == null) return;

            for (URI uri : cacheFiles) {
                Path p = new Path(uri.getPath());
                String name = p.getName();
                // read any file, but we expect "part-*" from degree output
                try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(name)))) {
                    String line;
                    while ((line = br.readLine()) != null) {
                        line = line.trim();
                        if (line.isEmpty()) continue;
                        String[] parts = line.split("\\s+");
                        if (parts.length < 2) continue;
                        String id = parts[0].trim();
                        String degStr = parts[1].trim();
                        try {
                            int d = Integer.parseInt(degStr);
                            degMap.put(id, d);
                        } catch (Exception ignored) {}
                    }
                }
            }
        }

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString().trim();
            if (line.isEmpty()) return;

            // pair \t common \t suppliersCSV
            String[] parts = line.split("\\s+");
            if (parts.length < 3) return;

            String pair = parts[0].trim();
            String commonStr = parts[1].trim();
            String suppliersCsv = parts[2].trim();

            int common;
            try {
                common = Integer.parseInt(commonStr);
            } catch (Exception e) {
                return;
            }

            String[] ab = pair.split(",");
            if (ab.length != 2) return;

            String a = ab[0];
            String b = ab[1];

            Integer degA = degMap.get(a);
            Integer degB = degMap.get(b);
            if (degA == null || degB == null) return;

            int union = degA + degB - common;
            double sim = (union <= 0) ? 0.0 : ((double) common) / ((double) union);
            if (sim <= 0.0) return;

            // emit to A: candidate B
            outKey.set(a);
            outVal.set(b + "\t" + sim + "\t" + suppliersCsv);
            context.write(outKey, outVal);

            // emit to B: candidate A
            outKey.set(b);
            outVal.set(a + "\t" + sim + "\t" + suppliersCsv);
            context.write(outKey, outVal);
        }
    }

    static class Candidate {
        String other;
        double sim;
        String suppliersCsv;

        Candidate(String other, double sim, String suppliersCsv) {
            this.other = other;
            this.sim = sim;
            this.suppliersCsv = suppliersCsv;
        }
    }

    public static class Job4Reducer extends Reducer<Text, Text, Text, NullWritable> {
        private final Text outLine = new Text();
        private int K;

        @Override
        protected void setup(Context context) {
            K = context.getConfiguration().getInt("taskE.topk", 4);
        }

        @Override
        protected void reduce(Text company, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {

            // Keep best K by (sim desc, otherId asc numeric)
            PriorityQueue<Candidate> pq = new PriorityQueue<>(K + 1, (x, y) -> {
                // min-heap: worst at top
                int c = Double.compare(x.sim, y.sim); // smaller sim is worse
                if (c != 0) return c;
                // if sim tie: larger id is worse (because we want smaller ids)
                return compareNumericId(y.other, x.other);
            });

            for (Text v : values) {
                String[] parts = v.toString().split("\\t");
                if (parts.length < 3) continue;
                String other = parts[0];
                double sim;
                try {
                    sim = Double.parseDouble(parts[1]);
                } catch (Exception e) {
                    continue;
                }
                String suppliersCsv = parts[2];

                Candidate cand = new Candidate(other, sim, suppliersCsv);

                if (pq.size() < K) {
                    pq.offer(cand);
                } else {
                    Candidate worst = pq.peek();
                    if (isBetter(cand, worst)) {
                        pq.poll();
                        pq.offer(cand);
                    }
                }
            }

            if (pq.isEmpty()) return;

            // Extract and sort best list by (sim desc, otherId asc numeric)
            ArrayList<Candidate> best = new ArrayList<>(pq);
            best.sort((x, y) -> {
                int c = Double.compare(y.sim, x.sim);
                if (c != 0) return c;
                return compareNumericId(x.other, y.other);
            });

            String a = company.toString();
            for (Candidate c : best) {
                // Format: A:B, {C,E}, simscore
                // suppliersCsv already comma-joined.
                String line = a + ":" + c.other + ", {" + c.suppliersCsv + "}, " + trimSim(c.sim);
                outLine.set(line);
                context.write(outLine, NullWritable.get());
            }
        }

        private static boolean isBetter(Candidate a, Candidate b) {
            int c = Double.compare(a.sim, b.sim);
            if (c != 0) return c > 0; // higher sim better
            // tie: smaller id better
            return compareNumericId(a.other, b.other) < 0;
        }

        private static String trimSim(double sim) {
            // keep a reasonable precision without scientific notation for small decimals
            // you can change to String.format(Locale.US, "%.6f", sim) if your TA prefers fixed decimals
            String s = Double.toString(sim);
            if (s.length() > 12) s = s.substring(0, 12);
            // remove trailing zeros
            if (s.contains(".")) {
                while (s.endsWith("0")) s = s.substring(0, s.length() - 1);
                if (s.endsWith(".")) s = s.substring(0, s.length() - 1);
            }
            return s;
        }
    }

    // -------------------- Driver --------------------

    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("Usage: TaskE <input_relation> <output_dir>");
            System.exit(2);
        }

        Configuration conf = new Configuration();

        // Compression of map outputs helps large dataset (hint in PDF)
        conf.setBoolean("mapreduce.map.output.compress", true);
        conf.set("mapreduce.map.output.compress.codec", "org.apache.hadoop.io.compress.SnappyCodec");

        // TopK = 4 for TaskE
        conf.setInt("taskE.topk", 4);

        Path input = new Path(args[0]);
        Path out = new Path(args[1]);

        Path tmp1 = new Path(args[1] + "_tmp1");
        Path tmp2 = new Path(args[1] + "_tmp2");
        Path tmp3 = new Path(args[1] + "_tmp3");
        Path tmpDeg = new Path(args[1] + "_deg");

        // ---------------- Job1 ----------------
        Job job1 = Job.getInstance(conf, "TaskE-Job1-DedupAndDegree");
        job1.setJarByClass(TaskE.class);

        job1.setMapperClass(Job1Mapper.class);
        job1.setReducerClass(Job1Reducer.class);

        job1.setMapOutputKeyClass(Text.class);
        job1.setMapOutputValueClass(Text.class);

        job1.setOutputKeyClass(Text.class);
        job1.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job1, input);
        FileOutputFormat.setOutputPath(job1, tmp1);

        // Two named outputs:
        MultipleOutputs.addNamedOutput(job1, "sb", TextOutputFormat.class, Text.class, Text.class);
        MultipleOutputs.addNamedOutput(job1, "deg", TextOutputFormat.class, Text.class, Text.class);

        // Reduce count: tune if needed
        // job1.setNumReduceTasks(80);

        if (!job1.waitForCompletion(true)) System.exit(1);

        // Move degree output to separate path for cache convenience
        // degree files are in tmp1/deg-m-???? or deg-r-???? depending on framework
        // We'll just use glob in your run script to add all deg files to cache.

        // ---------------- Job2 ----------------
        Job job2 = Job.getInstance(conf, "TaskE-Job2-InvertAndPairs");
        job2.setJarByClass(TaskE.class);

        job2.setMapperClass(Job2Mapper.class);
        job2.setReducerClass(Job2Reducer.class);

        job2.setMapOutputKeyClass(Text.class);
        job2.setMapOutputValueClass(Text.class);

        job2.setOutputKeyClass(Text.class);
        job2.setOutputValueClass(Text.class);

        // Input = Job1 "sb" named output folder
        // Hadoop writes named outputs under tmp1/sb-* (in practice "sb-m-00000" style).
        FileInputFormat.addInputPath(job2, new Path(tmp1, "sb-*"));
        FileOutputFormat.setOutputPath(job2, tmp2);

        // job2.setNumReduceTasks(120);

        if (!job2.waitForCompletion(true)) System.exit(1);

        // ---------------- Job3 ----------------
        Job job3 = Job.getInstance(conf, "TaskE-Job3-AggregateCommonSuppliers");
        job3.setJarByClass(TaskE.class);

        job3.setMapperClass(Job3Mapper.class);
        job3.setReducerClass(Job3Reducer.class);

        job3.setMapOutputKeyClass(Text.class);
        job3.setMapOutputValueClass(Text.class);

        job3.setOutputKeyClass(Text.class);
        job3.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job3, tmp2);
        FileOutputFormat.setOutputPath(job3, tmp3);

        // job3.setNumReduceTasks(120);

        if (!job3.waitForCompletion(true)) System.exit(1);

        // ---------------- Job4 ----------------
        Job job4 = Job.getInstance(conf, "TaskE-Job4-SimilarityTopK");
        job4.setJarByClass(TaskE.class);

        job4.setMapperClass(Job4Mapper.class);
        job4.setReducerClass(Job4Reducer.class);

        job4.setMapOutputKeyClass(Text.class);
        job4.setMapOutputValueClass(Text.class);

        job4.setOutputKeyClass(Text.class);
        job4.setOutputValueClass(NullWritable.class);

        FileInputFormat.addInputPath(job4, tmp3);
        FileOutputFormat.setOutputPath(job4, out);

        // Add ALL degree part files from Job1 "deg" output to DistributedCache.
        // Named output files pattern: tmp1/deg-* . We add them via -files in your run command (recommended).
        // Here we assume you will use -files option; so code doesn't hardcode cache URIs.

        // job4.setNumReduceTasks(80);

        if (!job4.waitForCompletion(true)) System.exit(1);
    }
}
