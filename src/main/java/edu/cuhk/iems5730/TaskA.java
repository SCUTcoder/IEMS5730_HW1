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

/**
 * TaskA - Find the pair of companies with the maximum number of common suppliers.
 *
 * Assumed input format: each line is an edge "company supplier" (space/tab/comma-separated).
 * Example:
 *   c1 s9
 *   c2 s9
 *   c1 s3
 *
 * Output format:
 *   <company1>\t<company2>\t<count>
 *
 * This implementation uses TWO MapReduce jobs:
 *   Job1: supplier -> list(companies) -> emit all pairs (c1,c2) with +1 per supplier; reduce sums.
 *   Job2: scan pair counts and pick the global maximum (single reducer, O(1) memory).
 */
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

            // key: supplier, val: company
            outKey.set(supplier);
            outVal.set(company);
            context.write(outKey, outVal);
        }
    }

    public static class SupplierToCompanyReducer extends Reducer<Text, Text, Text, IntWritable> {
        private final Text outKey = new Text();
        private static final IntWritable ONE = new IntWritable(1);

        // 为了防止极端 supplier（连接到成千上万公司）产生 O(k^2) 爆炸，
        // 这里提供一个可配置“上限”。默认 5000，你可以按数据调小/调大。
        // 如果你必须 100% 精确且数据确实有超级 hub supplier，就把它设得更大。
        private int maxCompaniesPerSupplier;

        @Override
        protected void setup(Context context) {
            Configuration conf = context.getConfiguration();
            maxCompaniesPerSupplier = conf.getInt("taskA.maxCompaniesPerSupplier", 5000);
        }

        @Override
        protected void reduce(Text supplier, Iterable<Text> companies, Context context)
                throws IOException, InterruptedException {

            // 去重：同一个 supplier 下可能出现重复 company 记录
            HashSet<String> uniq = new HashSet<>();
            for (Text c : companies) {
                uniq.add(c.toString());
                // 早停：避免 set 无限膨胀
                if (uniq.size() > maxCompaniesPerSupplier) {
                    // 超过上限：直接丢弃这个 supplier（防爆）
                    // 如果你想“尽量保留”，可以改成只保留前 maxCompaniesPerSupplier 个（但会影响精确性）
                    return;
                }
            }

            if (uniq.size() < 2) return;

            ArrayList<String> list = new ArrayList<>(uniq);
            Collections.sort(list);

            // 生成所有 company pairs: (ci,cj), i<j
            // 每出现一次 pair，表示它们共享了当前 supplier -> +1
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

    // Combiner/Reducer: sum counts per pair
    public static class SumIntReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        private final IntWritable outVal = new IntWritable();

        @Override
        protected void reduce(Text key, Iterable<IntWritable> vals, Context context)
                throws IOException, InterruptedException {
            long sum = 0;
            for (IntWritable v : vals) sum += v.get();
            // 计数通常不会超过 int 范围；如果你担心，改 LongWritable
            outVal.set((int) Math.min(sum, Integer.MAX_VALUE));
            context.write(key, outVal);
        }
    }

    // -------------------------
    // Job2: find global max pair count
    // -------------------------
    public static class MaxScanMapper extends Mapper<LongWritable, Text, Text, Text> {
        private static final Text CONST_KEY = new Text("MAX");
        private final Text outVal = new Text();

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            // line: "c1 \t c2 \t count"  (from Job1 output)
            String line = value.toString().trim();
            if (line.isEmpty()) return;

            String[] parts = line.split("\\t");
            if (parts.length < 3) return;

            String c1 = parts[0].trim();
            String c2 = parts[1].trim();
            String cntStr = parts[2].trim();
            if (c1.isEmpty() || c2.isEmpty() || cntStr.isEmpty()) return;

            // value: "count \t c1 \t c2" 方便 reducer 比较
            outVal.set(cntStr + "\t" + c1 + "\t" + c2);
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

                // 选更大；若相等，用字典序稳定输出（方便验收）
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

    // Ensure Job2 has only one reducer (global max)
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
        String tmp = output + "_job1_pairs";

        Configuration conf = getConf();

        // 强烈建议：避免 MR 自己把 reducer “600s 超时干掉”
        // 你之前日志就是 600s timeout -> taskTimedOut=true
        // 这里直接在代码里设 0（永不超时），保底能跑完。
        conf.setInt("mapreduce.task.timeout", 0);

        // 关闭推测执行，避免“重复 reducer + unknown container complete event”这种伴随现象更频繁
        conf.setBoolean("mapreduce.map.speculative", false);
        conf.setBoolean("mapreduce.reduce.speculative", false);

        // Job1: supplier -> company list -> company pairs
        Job job1 = Job.getInstance(conf, "TaskA-Job1-CountCommonSuppliers");
        job1.setJarByClass(TaskA.class);

        job1.setMapperClass(SupplierToCompanyMapper.class);
        job1.setMapOutputKeyClass(Text.class);
        job1.setMapOutputValueClass(Text.class);

        // reducer 生成 pair -> 1
        job1.setReducerClass(SupplierToCompanyReducer.class);
        job1.setOutputKeyClass(Text.class);
        job1.setOutputValueClass(IntWritable.class);

        // 这里 combiner 不能用 SupplierToCompanyReducer（类型不一致）
        // 但我们有第二阶段“SumIntReducer”才是加和逻辑，所以 Job1 只负责生成 pair->1
        // 为了减少网络传输，我们把 “pair->1 的加和” 放到 Job1 的后半：用第二个 job 更稳
        // ——因此：Job1 输出 pair->1，Job1 不设 combiner。

        FileInputFormat.addInputPath(job1, new Path(input));
        FileOutputFormat.setOutputPath(job1, new Path(tmp));

        // Ensure tmp output doesn't exist
        FileSystem fs = FileSystem.get(conf);
        fs.delete(new Path(tmp), true);

        boolean ok1 = job1.waitForCompletion(true);
        if (!ok1) return 1;

        // Job1.5: sum counts per pair（把 pair->1 变成 pair->count）
        String tmp2 = output + "_job1_counts";
        Job job1b = Job.getInstance(conf, "TaskA-Job1b-SumPairCounts");
        job1b.setJarByClass(TaskA.class);

        // Mapper: identity parse "pair\t1" -> (pair,1)
        job1b.setMapperClass(new Mapper<LongWritable, Text, Text, IntWritable>() {
            private final Text k = new Text();
            private final IntWritable v = new IntWritable();

            @Override
            protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
                String line = value.toString().trim();
                if (line.isEmpty()) return;
                String[] parts = line.split("\\t");
                if (parts.length < 3) return; // pair is "c1\tc2" then "\t1" => total >=3
                String c1 = parts[0].trim();
                String c2 = parts[1].trim();
                String cnt = parts[2].trim();
                if (c1.isEmpty() || c2.isEmpty() || cnt.isEmpty()) return;
                int n;
                try { n = Integer.parseInt(cnt); } catch (NumberFormatException e) { return; }
                k.set(c1 + "\t" + c2);
                v.set(n);
                context.write(k, v);
            }
        }.getClass());

        job1b.setMapOutputKeyClass(Text.class);
        job1b.setMapOutputValueClass(IntWritable.class);

        job1b.setCombinerClass(SumIntReducer.class);
        job1b.setReducerClass(SumIntReducer.class);

        job1b.setOutputKeyClass(Text.class);
        job1b.setOutputValueClass(IntWritable.class);

        FileInputFormat.addInputPath(job1b, new Path(tmp));
        FileOutputFormat.setOutputPath(job1b, new Path(tmp2));
        fs.delete(new Path(tmp2), true);

        boolean ok1b = job1b.waitForCompletion(true);
        if (!ok1b) return 1;

        // Job2: find global max (single reducer)
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
        fs.delete(new Path(output), true);

        boolean ok2 = job2.waitForCompletion(true);

        // 清理中间目录（可选）
        fs.delete(new Path(tmp), true);
        fs.delete(new Path(tmp2), true);

        return ok2 ? 0 : 1;
    }

    public static void main(String[] args) throws Exception {
        int res = ToolRunner.run(new Configuration(), new TaskA(), args);
        System.exit(res);
    }
}
