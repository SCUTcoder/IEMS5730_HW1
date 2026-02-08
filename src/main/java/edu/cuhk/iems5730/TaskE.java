package edu.cuhk.iems5730;

import java.io.*;
import java.math.BigInteger;
import java.net.URI;
import java.util.*;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.*;
import org.apache.hadoop.mapreduce.lib.output.*;
import org.apache.hadoop.mapreduce.lib.partition.HashPartitioner;

/**
 * Task E (Large dataset): For EVERY company, output TOP-K (K=4) most similar companies
 * and their common suppliers, format same as Task B:
 *
 * A:B, {C,E}, simscore
 *
 * Similarity(A,B) = |Sup(A) ∩ Sup(B)| / |Sup(A) ∪ Sup(B)|, if denom=0 then 0.
 *
 * Scalable design:
 *  Job1: build unique edges (buyer->supplier) and degree(buyer)=#distinct suppliers
 *  Job2: group by supplier to generate (pair -> supplier) for common suppliers
 *  Job3: aggregate per pair => list of common suppliers + count
 *  Job4: join degrees (cache) to compute similarity, use CompositeKey + Secondary Sort
 *        to output Top4 for each company.
 */
public class TaskE {

    // ---------------------------
    // Helpers: numeric-string compare (IDs might overflow long)
    // ---------------------------
    private static int compareNumericString(String a, String b) {
        // treat as non-negative integer strings
        a = stripLeadingZeros(a);
        b = stripLeadingZeros(b);
        if (a.length() != b.length()) return Integer.compare(a.length(), b.length());
        return a.compareTo(b);
    }

    private static String stripLeadingZeros(String s) {
        int i = 0;
        while (i < s.length() - 1 && s.charAt(i) == '0') i++;
        return s.substring(i);
    }

    // ---------------------------
    // Job1: buyer,supplier edges -> dedup + output degree
    // ---------------------------
    public static class EdgeMapper extends Mapper<LongWritable, Text, Text, Text> {
        private final Text k = new Text();
        private final Text v = new Text();

        @Override
        protected void map(LongWritable key, Text value, Context ctx) throws IOException, InterruptedException {
            String line = value.toString().trim();
            if (line.isEmpty()) return;
            String[] parts = line.split("\\s+");
            if (parts.length < 2) return;
            String buyer = parts[0];
            String supplier = parts[1];
            k.set(buyer);
            v.set(supplier);
            ctx.write(k, v);
        }
    }

    /**
     * Output:
     *  - Main output (edges): buyer \t supplier   (one line per unique edge)
     *  - Named output "deg": buyer \t degree
     */
    public static class EdgeDedupReducer extends Reducer<Text, Text, Text, Text> {
        private MultipleOutputs<Text, Text> mos;
        private final Text outVal = new Text();

        @Override
        protected void setup(Context ctx) {
            mos = new MultipleOutputs<>(ctx);
        }

        @Override
        protected void reduce(Text buyer, Iterable<Text> suppliers, Context ctx)
                throws IOException, InterruptedException {
            // dedup suppliers for buyer
            HashSet<String> set = new HashSet<>();
            for (Text t : suppliers) {
                String s = t.toString();
                if (!s.isEmpty()) set.add(s);
            }

            // write edges
            for (String s : set) {
                outVal.set(s);
                ctx.write(buyer, outVal);
            }

            // write degree
            mos.write("deg", buyer, new Text(Integer.toString(set.size())));
        }

        @Override
        protected void cleanup(Context ctx) throws IOException, InterruptedException {
            mos.close();
        }
    }

    // ---------------------------
    // Job2: invert edges by supplier => for each supplier, list buyers -> emit all pairs
    // ---------------------------
    public static class SupplierToBuyersMapper extends Mapper<LongWritable, Text, Text, Text> {
        private final Text k = new Text();
        private final Text v = new Text();

        @Override
        protected void map(LongWritable key, Text value, Context ctx) throws IOException, InterruptedException {
            // input: buyer \t supplier
            String line = value.toString().trim();
            if (line.isEmpty()) return;
            String[] parts = line.split("\\s+");
            if (parts.length < 2) return;
            String buyer = parts[0];
            String supplier = parts[1];
            k.set(supplier);
            v.set(buyer);
            ctx.write(k, v);
        }
    }

    public static class PairEmitReducer extends Reducer<Text, Text, Text, Text> {
        private final Text outK = new Text();
        private final Text outV = new Text();

        @Override
        protected void reduce(Text supplier, Iterable<Text> buyersIt, Context ctx)
                throws IOException, InterruptedException {

            // dedup buyers for this supplier
            ArrayList<String> buyers = new ArrayList<>();
            HashSet<String> seen = new HashSet<>();
            for (Text t : buyersIt) {
                String b = t.toString();
                if (b.isEmpty()) continue;
                if (seen.add(b)) buyers.add(b);
            }

            if (buyers.size() < 2) return;

            // sort buyers ascending (numeric string)
            buyers.sort(TaskE::compareNumericString);

            // emit all unordered pairs A,B (A<B) with value = supplier
            for (int i = 0; i < buyers.size(); i++) {
                for (int j = i + 1; j < buyers.size(); j++) {
                    String a = buyers.get(i);
                    String b = buyers.get(j);
                    outK.set(a + "," + b);
                    outV.set(supplier.toString());
                    ctx.write(outK, outV);
                }
            }
        }
    }

    // ---------------------------
    // Job3: aggregate per pair => common suppliers list + count
    // output: A,B \t count \t supplier1,supplier2,...
    // ---------------------------
    public static class PairAggReducer extends Reducer<Text, Text, Text, Text> {
        private final Text outVal = new Text();

        @Override
        protected void reduce(Text pair, Iterable<Text> suppliers, Context ctx)
                throws IOException, InterruptedException {
            // dedup suppliers for pair (safe)
            HashSet<String> set = new HashSet<>();
            for (Text t : suppliers) {
                String s = t.toString();
                if (!s.isEmpty()) set.add(s);
            }
            if (set.isEmpty()) return;

            ArrayList<String> list = new ArrayList<>(set);
            list.sort(TaskE::compareNumericString);

            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < list.size(); i++) {
                if (i > 0) sb.append(",");
                sb.append(list.get(i));
            }

            // value: count \t suppliersCSV
            outVal.set(set.size() + "\t" + sb);
            ctx.write(pair, outVal);
        }
    }

    // ---------------------------
    // Job4: compute similarity + secondary sort for TopK per company
    // Read degree file from DistributedCache (OK for ~100k companies)
    // Input: pair "A,B" \t count \t suppliersCSV
    // Emit for BOTH A and B, so every company gets candidates.
    // ---------------------------

    public static class TopKey implements WritableComparable<TopKey> {
        private Text company = new Text();
        private DoubleWritable negSim = new DoubleWritable(); // sort ascending => sim desc
        private Text partner = new Text();

        public TopKey() {}

        public TopKey(String company, double sim, String partner) {
            this.company.set(company);
            this.negSim.set(-sim);
            this.partner.set(partner);
        }

        public String getCompany() { return company.toString(); }
        public String getPartner() { return partner.toString(); }
        public double getSim() { return -negSim.get(); }

        @Override
        public void write(DataOutput out) throws IOException {
            company.write(out);
            negSim.write(out);
            partner.write(out);
        }

        @Override
        public void readFields(DataInput in) throws IOException {
            company.readFields(in);
            negSim.readFields(in);
            partner.readFields(in);
        }

        @Override
        public int compareTo(TopKey o) {
            int c = compareNumericString(this.company.toString(), o.company.toString());
            if (c != 0) return c;

            // negSim ascending (i.e., sim descending)
            c = Double.compare(this.negSim.get(), o.negSim.get());
            if (c != 0) return c;

            // partner ascending numeric
            return compareNumericString(this.partner.toString(), o.partner.toString());
        }

        @Override
        public int hashCode() {
            return company.hashCode();
        }

        @Override
        public boolean equals(Object obj) {
            if (!(obj instanceof TopKey)) return false;
            TopKey o = (TopKey) obj;
            return company.equals(o.company) && negSim.equals(o.negSim) && partner.equals(o.partner);
        }

        @Override
        public String toString() {
            return company + "\t" + (-negSim.get()) + "\t" + partner;
        }
    }

    public static class TopKeyPartitioner extends Partitioner<TopKey, Text> {
        private final HashPartitioner<Text, Text> hp = new HashPartitioner<>();
        @Override
        public int getPartition(TopKey key, Text value, int numPartitions) {
            return hp.getPartition(new Text(key.getCompany()), value, numPartitions);
        }
    }

    public static class GroupByCompanyComparator extends WritableComparator {
        protected GroupByCompanyComparator() {
            super(TopKey.class, true);
        }

        @Override
        public int compare(WritableComparable a, WritableComparable b) {
            TopKey k1 = (TopKey) a;
            TopKey k2 = (TopKey) b;
            return compareNumericString(k1.getCompany(), k2.getCompany());
        }
    }

    public static class SimilarityMapper extends Mapper<LongWritable, Text, TopKey, Text> {
        private final Map<String, Integer> degMap = new HashMap<>();
        private final Text outV = new Text();

        @Override
        protected void setup(Context ctx) throws IOException {
            // degrees file from cache: buyer \t degree
            URI[] cacheFiles = ctx.getCacheFiles();
            if (cacheFiles == null || cacheFiles.length == 0) {
                throw new IOException("Degree file not found in DistributedCache.");
            }

            // try each cache file; parse all lines
            for (URI u : cacheFiles) {
                File f = new File(new Path(u.getPath()).getName()); // localized name
                if (!f.exists()) {
                    // fallback: direct path
                    f = new File(u.getPath());
                }
                if (!f.exists()) continue;

                try (BufferedReader br = new BufferedReader(new FileReader(f))) {
                    String line;
                    while ((line = br.readLine()) != null) {
                        line = line.trim();
                        if (line.isEmpty()) continue;
                        String[] p = line.split("\\s+");
                        if (p.length < 2) continue;
                        String id = p[0];
                        int d;
                        try {
                            d = Integer.parseInt(p[1]);
                        } catch (NumberFormatException e) {
                            continue;
                        }
                        degMap.put(id, d);
                    }
                }
            }

            if (degMap.isEmpty()) {
                throw new IOException("Loaded degree map is empty. Check cache file content.");
            }
        }

        @Override
        protected void map(LongWritable key, Text value, Context ctx) throws IOException, InterruptedException {
            // input: A,B \t commonCount \t suppliersCSV
            String line = value.toString().trim();
            if (line.isEmpty()) return;

            String[] parts = line.split("\\t");
            if (parts.length < 3) return;

            String pair = parts[0];
            String[] ab = pair.split(",");
            if (ab.length != 2) return;

            String a = ab[0];
            String b = ab[1];

            int common;
            try {
                common = Integer.parseInt(parts[1]);
            } catch (NumberFormatException e) {
                return;
            }

            Integer da = degMap.get(a);
            Integer db = degMap.get(b);
            if (da == null) da = 0;
            if (db == null) db = 0;

            int denom = da + db - common;
            double sim = (denom <= 0) ? 0.0 : ((double) common / (double) denom);
            if (sim <= 0.0) return;

            String suppliersCSV = parts[2];

            // emit for company a: partner b
            outV.set(b + "\t" + suppliersCSV + "\t" + sim);
            ctx.write(new TopKey(a, sim, b), outV);

            // emit for company b: partner a (same sim, same suppliers list)
            outV.set(a + "\t" + suppliersCSV + "\t" + sim);
            ctx.write(new TopKey(b, sim, a), outV);
        }
    }

    public static class TopKReducer extends Reducer<TopKey, Text, Text, Text> {
        private static final int K = 4;
        private final Text outK = new Text();
        private final Text outV = new Text();

        @Override
        protected void reduce(TopKey key, Iterable<Text> vals, Context ctx)
                throws IOException, InterruptedException {

            String company = key.getCompany();
            outK.set(company);

            int kept = 0;
            // Because of secondary sort, vals are already in sim desc then partner asc order
            for (Text t : vals) {
                if (kept >= K) break;

                String[] p = t.toString().split("\\t", 3);
                if (p.length < 3) continue;

                String partner = p[0];
                String suppliersCSV = p[1];
                String simStr = p[2];

                // format: A:B, {C,E}, simscore
                String formattedSuppliers = "{" + suppliersCSV + "}";
                outV.set(company + ":" + partner + ", " + formattedSuppliers + ", " + simStr);
                ctx.write(outK, outV);
                kept++;
            }
        }
    }

    // ---------------------------
    // Main: run 4 jobs
    // args: <input_relation_path> <output_path>
    // ---------------------------
    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: TaskE <input_relation> <output_dir>");
            System.exit(1);
        }

        String input = args[0];
        String out = args[1];

        String tmp1 = out + "_tmp1_edges_deg";
        String tmp2 = out + "_tmp2_pair_by_supplier";
        String tmp3 = out + "_tmp3_pair_agg";
        String finalOut = out;

        Configuration conf = new Configuration();

        // ---------------- Job1 ----------------
        Job job1 = Job.getInstance(conf, "TaskE-Job1-EdgesAndDegree");
        job1.setJarByClass(TaskE.class);
        job1.setMapperClass(EdgeMapper.class);
        job1.setReducerClass(EdgeDedupReducer.class);

        job1.setMapOutputKeyClass(Text.class);
        job1.setMapOutputValueClass(Text.class);
        job1.setOutputKeyClass(Text.class);
        job1.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job1, new Path(input));
        FileOutputFormat.setOutputPath(job1, new Path(tmp1));

        // multiple outputs for degree
        MultipleOutputs.addNamedOutput(job1, "deg", TextOutputFormat.class, Text.class, Text.class);

        // optional compression for large
        job1.getConfiguration().setBoolean("mapreduce.map.output.compress", true);

        if (!job1.waitForCompletion(true)) System.exit(2);

        // degree file path (glob): tmp1/deg-m-00000 or deg-r-00000 etc.
        // We'll pass the whole "deg" directory to cache via a concrete file:
        // For safety, assume reducers => part-r-00000 under tmp1/deg/
        Path degFile = new Path(tmp1 + "/deg/part-r-00000");
        // If the framework emitted mapper-only named outputs, it might be part-m-00000.
        // We'll add both candidates to cache; the mapper will parse whichever exists.
        Path degFileM = new Path(tmp1 + "/deg/part-m-00000");

        // ---------------- Job2 ----------------
        Job job2 = Job.getInstance(conf, "TaskE-Job2-PairsBySupplier");
        job2.setJarByClass(TaskE.class);
        job2.setMapperClass(SupplierToBuyersMapper.class);
        job2.setReducerClass(PairEmitReducer.class);

        job2.setMapOutputKeyClass(Text.class);
        job2.setMapOutputValueClass(Text.class);
        job2.setOutputKeyClass(Text.class);
        job2.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job2, new Path(tmp1)); // reads edges from main output
        FileOutputFormat.setOutputPath(job2, new Path(tmp2));

        job2.getConfiguration().setBoolean("mapreduce.map.output.compress", true);

        if (!job2.waitForCompletion(true)) System.exit(3);

        // ---------------- Job3 ----------------
        Job job3 = Job.getInstance(conf, "TaskE-Job3-AggregatePairs");
        job3.setJarByClass(TaskE.class);

        job3.setMapperClass(Mapper.class); // identity mapper
        job3.setReducerClass(PairAggReducer.class);

        job3.setMapOutputKeyClass(Text.class);
        job3.setMapOutputValueClass(Text.class);
        job3.setOutputKeyClass(Text.class);
        job3.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job3, new Path(tmp2));
        FileOutputFormat.setOutputPath(job3, new Path(tmp3));

        job3.getConfiguration().setBoolean("mapreduce.map.output.compress", true);

        if (!job3.waitForCompletion(true)) System.exit(4);

        // ---------------- Job4 ----------------
        Job job4 = Job.getInstance(conf, "TaskE-Job4-TopK-SecondarySort");
        job4.setJarByClass(TaskE.class);

        job4.setMapperClass(SimilarityMapper.class);
        job4.setReducerClass(TopKReducer.class);

        job4.setMapOutputKeyClass(TopKey.class);
        job4.setMapOutputValueClass(Text.class);
        job4.setOutputKeyClass(Text.class);
        job4.setOutputValueClass(Text.class);

        job4.setPartitionerClass(TopKeyPartitioner.class);
        job4.setGroupingComparatorClass(GroupByCompanyComparator.class);

        FileInputFormat.addInputPath(job4, new Path(tmp3));
        FileOutputFormat.setOutputPath(job4, new Path(finalOut));

        // add degree cache (try both)
        job4.addCacheFile(degFile.toUri());
        job4.addCacheFile(degFileM.toUri());

        if (!job4.waitForCompletion(true)) System.exit(5);

        System.exit(0);
    }
}
