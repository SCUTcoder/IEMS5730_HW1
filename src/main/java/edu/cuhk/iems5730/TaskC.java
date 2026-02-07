package edu.cuhk.iems5730;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.IOException;

/**
 * Task C (Skew-safe):
 * For each community, count how many unique members are "common suppliers" of other companies.
 *
 * Common supplier definition (equivalent, scalable):
 * A company s is a common supplier if it supplies to >= 2 different buyers.
 *
 * Jobs:
 * Job1: Salted pre-aggregation by (supplier#salt) -> emits supplier \t MULTI or supplier \t ONE:buyer
 * Job2: Merge shards by supplier -> emits common supplier list (supplier)
 * Job3: Join common suppliers with labels -> emits Community k \t 1
 * Job4: Sum counts by community
 */
public class TaskC {

    // Tune this for skew handling: more buckets -> more parallelism for hot suppliers
    private static final String CONF_SALT_BUCKETS = "taskc.salt.buckets";
    private static final int DEFAULT_SALT_BUCKETS = 64;

    // =======================
    // Job1: Salted Pre-Agg
    // =======================

    /**
     * Input: relation file lines: buyer supplier
     * Output key: supplier#salt
     * Output value: buyer
     */
    public static class SaltedRelationMapper extends Mapper<LongWritable, Text, Text, Text> {
        private final Text outKey = new Text();
        private final Text outVal = new Text();
        private int saltBuckets;

        @Override
        protected void setup(Context context) {
            saltBuckets = context.getConfiguration().getInt(CONF_SALT_BUCKETS, DEFAULT_SALT_BUCKETS);
            if (saltBuckets <= 0) saltBuckets = DEFAULT_SALT_BUCKETS;
        }

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString().trim();
            if (line.isEmpty()) return;

            String[] parts = line.split("\\s+");
            if (parts.length < 2) return;

            String buyer = parts[0];
            String supplier = parts[1];

            // salt by buyer to spread hot supplier across reducers
            int salt = (buyer.hashCode() & Integer.MAX_VALUE) % saltBuckets;

            outKey.set(supplier + "#" + salt);
            outVal.set(buyer);
            context.write(outKey, outVal);
        }
    }

    /**
     * Partition by supplier only (ignore salt) OR by full key?
     * Here we partition by full key (supplier#salt) to spread load evenly.
     * (Default hash partitioner would also do this, but we keep it explicit.)
     */
    public static class FullKeyPartitioner extends Partitioner<Text, Text> {
        @Override
        public int getPartition(Text key, Text value, int numPartitions) {
            return (key.hashCode() & Integer.MAX_VALUE) % numPartitions;
        }
    }

    /**
     * Reducer for (supplier#salt) -> buyers
     * Emits:
     *   supplier \t MULTI        if >=2 distinct buyers within this shard
     *   supplier \t ONE:<buyer>  if only 1 distinct buyer within this shard
     *
     * NOTE: O(1) memory: we do NOT store all buyers, we stop once we find 2 unique.
     */
    public static class SaltedPreAggReducer extends Reducer<Text, Text, Text, Text> {
        private final Text outKey = new Text();
        private final Text outVal = new Text();

        @Override
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            String k = key.toString();
            int idx = k.lastIndexOf('#');
            if (idx <= 0) return;
            String supplier = k.substring(0, idx);

            String firstBuyer = null;
            boolean multi = false;

            for (Text v : values) {
                String buyer = v.toString();
                if (firstBuyer == null) {
                    firstBuyer = buyer;
                } else if (!firstBuyer.equals(buyer)) {
                    multi = true;
                    break; // early stop
                }
            }

            outKey.set(supplier);
            if (multi) {
                outVal.set("MULTI");
            } else if (firstBuyer != null) {
                outVal.set("ONE:" + firstBuyer);
            } else {
                return;
            }
            context.write(outKey, outVal);
        }
    }

    // =======================
    // Job2: Merge Shards
    // =======================

    /**
     * Input: supplier \t MULTI  OR supplier \t ONE:buyer
     * Output key: supplier
     * Output value: tag
     */
    public static class MergeMapper extends Mapper<LongWritable, Text, Text, Text> {
        private final Text outKey = new Text();
        private final Text outVal = new Text();

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString().trim();
            if (line.isEmpty()) return;

            String[] parts = line.split("\\t");
            if (parts.length < 2) return;

            outKey.set(parts[0]);
            outVal.set(parts[1]);
            context.write(outKey, outVal);
        }
    }

    /**
     * Reducer:
     * supplier is common if:
     *   - any shard says MULTI
     *   - OR there exist two different ONE:<buyer> across shards
     * Output: supplier \t (empty)
     */
    public static class MergeReducer extends Reducer<Text, Text, Text, Text> {
        private static final Text EMPTY = new Text("");

        @Override
        public void reduce(Text supplier, Iterable<Text> tags, Context context) throws IOException, InterruptedException {
            boolean isCommon = false;
            String oneBuyer = null;

            for (Text t : tags) {
                String s = t.toString();
                if ("MULTI".equals(s)) {
                    isCommon = true;
                    break;
                }
                if (s.startsWith("ONE:")) {
                    String buyer = s.substring(4);
                    if (oneBuyer == null) {
                        oneBuyer = buyer;
                    } else if (!oneBuyer.equals(buyer)) {
                        isCommon = true;
                        break;
                    }
                }
            }

            if (isCommon) {
                context.write(supplier, EMPTY); // emit supplier once
            }
        }
    }

    // =======================
    // Job3: Join with Labels
    // =======================

    // label file: companyId label
    public static class LabelMapper extends Mapper<LongWritable, Text, Text, Text> {
        private final Text outKey = new Text();
        private final Text outVal = new Text();

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString().trim();
            if (line.isEmpty()) return;

            String[] parts = line.split("\\s+");
            if (parts.length < 2) return;

            outKey.set(parts[0]);
            outVal.set("LABEL:" + parts[1]);
            context.write(outKey, outVal);
        }
    }

    // common supplier list: supplier \t ...
    public static class CommonSupplierMapper extends Mapper<LongWritable, Text, Text, Text> {
        private final Text outKey = new Text();
        private final Text outVal = new Text("CS");

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString().trim();
            if (line.isEmpty()) return;

            // could be "supplier" or "supplier\t..."
            String[] parts = line.split("\\t");
            if (parts.length < 1) return;

            outKey.set(parts[0]);
            context.write(outKey, outVal);
        }
    }

    /**
     * Join reducer:
     * If key(companyId) is both labeled and in common supplier list -> emit Community label \t 1
     */
    public static class JoinReducer extends Reducer<Text, Text, Text, IntWritable> {
        private static final IntWritable ONE = new IntWritable(1);
        private final Text outKey = new Text();

        @Override
        public void reduce(Text companyId, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            String label = null;
            boolean isCommonSupplier = false;

            for (Text v : values) {
                String s = v.toString();
                if (s.startsWith("LABEL:")) {
                    label = s.substring(6);
                } else if ("CS".equals(s)) {
                    isCommonSupplier = true;
                }
            }

            if (label != null && isCommonSupplier) {
                outKey.set("Community " + label);
                context.write(outKey, ONE);
            }
        }
    }

    // =======================
    // Job4: Sum by Community
    // =======================

    public static class SumMapper extends Mapper<LongWritable, Text, Text, IntWritable> {
        private final Text outKey = new Text();
        private final IntWritable outVal = new IntWritable();

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString().trim();
            if (line.isEmpty()) return;

            String[] parts = line.split("\\t");
            if (parts.length < 2) return;

            outKey.set(parts[0]);
            try {
                outVal.set(Integer.parseInt(parts[1]));
            } catch (NumberFormatException e) {
                return;
            }
            context.write(outKey, outVal);
        }
    }

    public static class SumReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        private final IntWritable outVal = new IntWritable();

        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable v : values) sum += v.get();
            outVal.set(sum);
            context.write(key, outVal);
        }
    }

    // =======================
    // Driver
    // =======================

    public static void main(String[] args) throws Exception {
        if (args.length != 3) {
            System.err.println("Usage: TaskC <relation input> <label input> <output path>");
            System.exit(1);
        }

        String relationInput = args[0];
        String labelInput = args[1];
        String outBase = args[2];

        Configuration conf = new Configuration();
        conf.setInt(CONF_SALT_BUCKETS, DEFAULT_SALT_BUCKETS);

        // ---------- Job1 ----------
        Job job1 = Job.getInstance(conf, "TaskC-Job1-SaltedPreAgg");
        job1.setJarByClass(TaskC.class);

        job1.setMapperClass(SaltedRelationMapper.class);
        job1.setReducerClass(SaltedPreAggReducer.class);

        job1.setMapOutputKeyClass(Text.class);
        job1.setMapOutputValueClass(Text.class);

        job1.setOutputKeyClass(Text.class);
        job1.setOutputValueClass(Text.class);

        // important: multiple reducers (avoid single reducer)
        job1.setNumReduceTasks(32);
        job1.setPartitionerClass(FullKeyPartitioner.class);

        TextInputFormat.addInputPath(job1, new Path(relationInput));
        Path job1Out = new Path(outBase + "_job1");
        FileOutputFormat.setOutputPath(job1, job1Out);

        if (!job1.waitForCompletion(true)) System.exit(1);

        // ---------- Job2 ----------
        Job job2 = Job.getInstance(conf, "TaskC-Job2-MergeShards");
        job2.setJarByClass(TaskC.class);

        job2.setMapperClass(MergeMapper.class);
        job2.setReducerClass(MergeReducer.class);

        job2.setMapOutputKeyClass(Text.class);
        job2.setMapOutputValueClass(Text.class);

        job2.setOutputKeyClass(Text.class);
        job2.setOutputValueClass(Text.class);

        job2.setNumReduceTasks(32);

        TextInputFormat.addInputPath(job2, job1Out);
        Path job2Out = new Path(outBase + "_job2_common_suppliers");
        FileOutputFormat.setOutputPath(job2, job2Out);

        if (!job2.waitForCompletion(true)) System.exit(1);

        // ---------- Job3 ----------
        Job job3 = Job.getInstance(conf, "TaskC-Job3-JoinLabels");
        job3.setJarByClass(TaskC.class);

        MultipleInputs.addInputPath(job3, new Path(labelInput), TextInputFormat.class, LabelMapper.class);
        MultipleInputs.addInputPath(job3, job2Out, TextInputFormat.class, CommonSupplierMapper.class);

        job3.setReducerClass(JoinReducer.class);

        job3.setMapOutputKeyClass(Text.class);
        job3.setMapOutputValueClass(Text.class);

        job3.setOutputKeyClass(Text.class);
        job3.setOutputValueClass(IntWritable.class);

        job3.setNumReduceTasks(16);

        Path job3Out = new Path(outBase + "_job3_joined");
        FileOutputFormat.setOutputPath(job3, job3Out);

        if (!job3.waitForCompletion(true)) System.exit(1);

        // ---------- Job4 ----------
        Job job4 = Job.getInstance(conf, "TaskC-Job4-SumByCommunity");
        job4.setJarByClass(TaskC.class);

        job4.setMapperClass(SumMapper.class);
        job4.setReducerClass(SumReducer.class);

        job4.setMapOutputKeyClass(Text.class);
        job4.setMapOutputValueClass(IntWritable.class);

        job4.setOutputKeyClass(Text.class);
        job4.setOutputValueClass(IntWritable.class);

        // Combiner reduces shuffle for counts
        job4.setCombinerClass(SumReducer.class);
        job4.setNumReduceTasks(8);

        TextInputFormat.addInputPath(job4, job3Out);
        FileOutputFormat.setOutputPath(job4, new Path(outBase));

        System.exit(job4.waitForCompletion(true) ? 0 : 1);
    }
}

