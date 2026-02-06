package edu.cuhk.iems5730;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;

import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;

import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;

public class TaskA extends Configured implements Tool {

    // -------------------------
    // Job1: supplier -> companies, generate pairs
    // -------------------------
    public static class SupplierToCompanyMapper extends Mapper<LongWritable, Text, Text, Text> {
        private final Text outKey = new Text();
        private final Text outVal = new Text();

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString().trim();
            if (line.isEmpty()) return;

            // split by tab / space / comma
            String[] parts = line.split("[\\s,]+");
            if (parts.length < 2) return;

            String company = parts[0].trim();
            String supplier = parts[1].trim();
            if (company.isEmpty() || supplier.isEmpty()) return;

            outKey.set(supplier);
            outVal.set(company);
            context.write(outKey, outVal);
        }
    }

    public static class SupplierToPairsReducer extends Reducer<Text, Text, Text, IntWritable> {
        private final Text outKey = new Text();
        private static final IntWritable ONE = new IntWritable(1);
        private int maxCompaniesPerSupplier;

        @Override
        protected void setup(Context context) {
            Configuration conf = context.getConfiguration();
            maxCompaniesPerSupplier = conf.getInt("taskA.maxCompaniesPerSupplier", 5000);
        }

        @Override
        protected void reduce(Text supplier, Iterable<Text> companies, Context context)
                throws IOException, InterruptedException {

            HashSet<String> uniq = new HashSet<>();
            for (Text c : companies) {
                uniq.add(c.toString());
                if (uniq.size() > maxCompaniesPerSupplier) {
                    // 防止超级 supplier 产生 O(k^2) 爆炸
                    return;
                }
            }
            if (uniq.size() < 2) return;

            ArrayList<String> list = new ArrayList<>(uniq);
            Collections.sort(list);

            int n = list.size();
            for (int i = 0; i < n; i++) {
                String ci = list.get(i);
                for (int j = i + 1; j < n; j++) {
                    String cj = list.get(j);
                    outKey.set(ci + "\t" + cj);
                    context.write(outKey, ONE);
                }
            }
        }
    }

    // -------------------------
    // Job1b: parse (c1 c2 1) and sum to (c1 c2 count)
    // -------------------------
    public static class PairOneMapper extends Mapper<LongWritable, Text, Text, IntWritable> {
        private final Text outKey = new Text();
        private final IntWritable outVal = new IntWritable();

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            // expected: c1 \t c2 \t 1
            String line = value.toString().trim();
            if (line.isEmpty()) return;

            String[] parts = line.split("\\t");
            if (parts.length < 3) return;

            String c1 = parts[0].trim();
            String c2 = parts[1].trim();
            String one = parts[2].trim();
            if (c1.isEmpty() || c2.isEmpty() || one.isEmpty()) return;

            int v;
            try {
                v = Integer.parseInt(one);
            } catch (NumberFormatException e) {
                return;
            }

            outKey.set(c1 + "\t" + c2);
            outVal.set(v);
            context.write(outKey, outVal);
        }
    }

    public static class SumIntReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        private final IntWritable outVal = new IntWritable();

        @Override
        protected void reduce(Text key, Iterable<IntWritable> vals, Context context)
                throws IOException, InterruptedException {
            long sum = 0;
            for (IntWritable v : vals) sum += v.get();
            outVal.set((int) Math.min(sum, Integer.MAX_VALUE));
            context.write(key, outVal);
        }
    }

    // -------------------------
    // Job2: find global max
    // -------------------------
    public static class MaxScanMapper extends Mapper<LongWritable, Text, Text, Text> {
        private static final Text CONST_KEY = new Text("MAX");
        private final Text outVal = new Text();

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            // expected: c1 \t c2 \t count
            String line = value.toString().trim();
            if (line.isEmpty()) return;

            String[] parts = line.split("\\t");
            if (parts.length < 3) return;

            String c1 = parts[0].trim();
            String c2 = parts[1].trim();
            String cnt = parts[2].trim();
            if (c1.isEmpty() || c2.isEmpty() || cnt.isEmpty()) return;

            // store as: count \t c1 \t c2
            outVal.set(cnt + "\t" + c1 + "\t" + c2);
            context.write(CONST_KEY, outVal);
        }
    }

    public static class MaxScanReducer extends Reducer<Text, Text, Text, IntWritable> {
        private final Text outKey = new Text();
        private final IntWritable outVal = new IntWritable();

        @Override
        protected void reduce(Text key, Iterable<Text> vals, Context context)
                throws IOException, InterruptedException {
            long best = -1;
            String bestC1 = "";
            String bestC2 = "";

            for (Text v : vals) {
                String[] parts = v.toString().split("\\t");
                if (parts.length < 3) continue;

                long cnt;
                try {
                    cnt = Long.parseLong(parts[0]);
                } catch (NumberFormatException e) {
                    continue;
                }

                String c1 = parts[1];
                String c2 = parts[2];

                if (cnt > best) {
                    best = cnt;
                    bestC1 = c1;
                    bestC2 = c2;
                } else if (cnt == best) {
                    String cur = c1 + "\t" + c2;
                    String prev = bestC1 + "\t" + bestC2;
                    if (cur.compareTo(prev) < 0) {
                        bestC1 = c1;
                        bestC2 = c2;
                    }
                }
            }

            if (best >= 0) {
                outKey.set(bestC1 + "\t" + bestC2);
                outVal.set((int) Math.min(best, Integer.MAX_VALUE));
                context.write(outKey, outVal);
            }
        }
    }

    public static class SinglePartitioner extends Partitioner<Text, Text> {
        @Override
        public int getPartition(Text key, Text value, int numPartitions) {
            return 0;
        }
    }

    // -------------------------
    // Driver
    // -------------------------
    @Override
    public int run(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: TaskA <input_path> <output_path>");
            return 2;
        }

        String input = args[0];
        String output = args[1];
        String tmp1 = output + "_job1_pairs";
        String tmp2 = output + "_job1_counts";

        Configuration conf = getConf();

        // fix your previous symptom: reducer task timeout at 600s
        conf.setInt("mapreduce.task.timeout", 0);
        conf.setBoolean("mapreduce.map.speculative", false);
        conf.setBoolean("mapreduce.reduce.speculative", false);

        FileSystem fs = FileSystem.get(conf);
        fs.delete(new Path(tmp1), true);
        fs.delete(new Path(tmp2), true);
        fs.delete(new Path(output), true);

        // Job1
        Job job1 = Job.getInstance(conf, "TaskA-Job1-GeneratePairs");
        job1.setJarByClass(TaskA.class);

        job1.setMapperClass(SupplierToCompanyMapper.class);
        job1.setMapOutputKeyClass(Text.class);
        job1.setMapOutputValueClass(Text.class);

        job1.setReducerClass(SupplierToPairsReducer.class);
        job1.setOutputKeyClass(Text.class);
        job1.setOutputValueClass(IntWritable.class);

        FileInputFormat.addInputPath(job1, new Path(input));
        FileOutputFormat.setOutputPath(job1, new Path(tmp1));

        if (!job1.waitForCompletion(true)) return 1;

        // Job1b (sum)
        Job job1b = Job.getInstance(conf, "TaskA-Job1b-SumPairCounts");
        job1b.setJarByClass(TaskA.class);

        job1b.setMapperClass(PairOneMapper.class);
        job1b.setMapOutputKeyClass(Text.class);
        job1b.setMapOutputValueClass(IntWritable.class);

        job1b.setCombinerClass(SumIntReducer.class);
        job1b.setReducerClass(SumIntReducer.class);

        job1b.setOutputKeyClass(Text.class);
        job1b.setOutputValueClass(IntWritable.class);

        FileInputFormat.addInputPath(job1b, new Path(tmp1));
        FileOutputFormat.setOutputPath(job1b, new Path(tmp2));

        if (!job1b.waitForCompletion(true)) return 1;

        // Job2 (global max)
        Job job2 = Job.getInstance(conf, "TaskA-Job2-FindGlobalMax");
        job2.setJarByClass(TaskA.class);

        job2.setMapperClass(MaxScanMapper.class);
        job2.setMapOutputKeyClass(Text.class);
        job2.setMapOutputValueClass(Text.class);

        job2.setPartitionerClass(SinglePartitioner.class);
        job2.setNumReduceTasks(1);

        job2.setReducerClass(MaxScanReducer.class);
        job2.setOutputKeyClass(Text.class);
        job2.setOutputValueClass(IntWritable.class);

        FileInputFormat.addInputPath(job2, new Path(tmp2));
        FileOutputFormat.setOutputPath(job2, new Path(output));

        boolean ok = job2.waitForCompletion(true);

        // cleanup temp
        fs.delete(new Path(tmp1), true);
        fs.delete(new Path(tmp2), true);

        return ok ? 0 : 1;
    }

    public static void main(String[] args) throws Exception {
        int res = ToolRunner.run(new Configuration(), new TaskA(), args);
        System.exit(res);
    }
}
