package edu.cuhk.iems5730;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.IOException;
import java.util.*;

/**
 * Task C: For each community in the medium dataset, figure out how many (unique) members
 * act as the common suppliers of other companies.
 * 
 * This requires 3 MapReduce jobs:
 * Job1: Find all common suppliers across all company pairs
 * Job2: Join with labels to get community information
 * Job3: Count unique members per community
 */
public class TaskC {

    // ========== Job 1: Build Supplier List ==========
    
    public static class SupplierListMapper extends Mapper<Object, Text, Text, Text> {
        private Text buyer = new Text();
        private Text supplier = new Text();

        @Override
        public void map(Object key, Text value, Context context) 
                throws IOException, InterruptedException {
            String line = value.toString().trim();
            if (line.isEmpty()) return;
            
            String[] parts = line.split("\\s+");
            if (parts.length >= 2) {
                buyer.set(parts[0]);
                supplier.set(parts[1]);
                context.write(buyer, supplier);
            }
        }
    }

    public static class SupplierListReducer extends Reducer<Text, Text, Text, Text> {
        @Override
        public void reduce(Text key, Iterable<Text> values, Context context) 
                throws IOException, InterruptedException {
            Set<String> suppliers = new TreeSet<>();
            for (Text val : values) {
                suppliers.add(val.toString());
            }
            
            if (!suppliers.isEmpty()) {
                context.write(key, new Text(String.join(",", suppliers)));
            }
        }
    }

    // ========== Job 2: Find All Common Suppliers ==========
    
    public static class CommonSupplierMapper extends Mapper<Object, Text, Text, Text> {
        @Override
        public void map(Object key, Text value, Context context) 
                throws IOException, InterruptedException {
            String line = value.toString().trim();
            if (line.isEmpty()) return;
            
            String[] parts = line.split("\\t");
            if (parts.length >= 2) {
                context.write(new Text("ALL"), new Text(parts[0] + ":" + parts[1]));
            }
        }
    }

    public static class CommonSupplierReducer extends Reducer<Text, Text, Text, Text> {
        @Override
        public void reduce(Text key, Iterable<Text> values, Context context) 
                throws IOException, InterruptedException {
            
            List<CompanySuppliers> companies = new ArrayList<>();
            for (Text val : values) {
                String[] parts = val.toString().split(":", 2);
                if (parts.length == 2) {
                    String companyId = parts[0];
                    Set<String> suppliers = new HashSet<>(Arrays.asList(parts[1].split(",")));
                    companies.add(new CompanySuppliers(companyId, suppliers));
                }
            }
            
            // Find all common suppliers (unique)
            Set<String> allCommonSuppliers = new HashSet<>();
            
            for (int i = 0; i < companies.size(); i++) {
                for (int j = i + 1; j < companies.size(); j++) {
                    Set<String> common = new HashSet<>(companies.get(i).suppliers);
                    common.retainAll(companies.get(j).suppliers);
                    allCommonSuppliers.addAll(common);
                }
            }
            
            // Output each common supplier
            for (String supplier : allCommonSuppliers) {
                context.write(new Text(supplier), new Text(""));
            }
        }
        
        private static class CompanySuppliers {
            String companyId;
            Set<String> suppliers;
            
            CompanySuppliers(String id, Set<String> sup) {
                this.companyId = id;
                this.suppliers = sup;
            }
        }
    }

    // ========== Job 3: Join with Labels and Count by Community ==========
    
    // Mapper for label file
    public static class LabelMapper extends Mapper<Object, Text, Text, Text> {
        @Override
        public void map(Object key, Text value, Context context) 
                throws IOException, InterruptedException {
            String line = value.toString().trim();
            if (line.isEmpty()) return;
            
            String[] parts = line.split("\\s+");
            if (parts.length >= 2) {
                String companyId = parts[0];
                String label = parts[1];
                context.write(new Text(companyId), new Text("LABEL:" + label));
            }
        }
    }
    
    // Mapper for common supplier list
    public static class SupplierMapper extends Mapper<Object, Text, Text, Text> {
        @Override
        public void map(Object key, Text value, Context context) 
                throws IOException, InterruptedException {
            String line = value.toString().trim();
            if (line.isEmpty()) return;
            
            String[] parts = line.split("\\t");
            if (parts.length >= 1) {
                String supplierId = parts[0];
                context.write(new Text(supplierId), new Text("SUPPLIER"));
            }
        }
    }
    
    public static class JoinReducer extends Reducer<Text, Text, Text, IntWritable> {
        @Override
        public void reduce(Text key, Iterable<Text> values, Context context) 
                throws IOException, InterruptedException {
            
            String label = null;
            boolean isCommonSupplier = false;
            
            for (Text val : values) {
                String value = val.toString();
                if (value.startsWith("LABEL:")) {
                    label = value.substring(6);
                } else if (value.equals("SUPPLIER")) {
                    isCommonSupplier = true;
                }
            }
            
            // Only count if this company is both labeled and a common supplier
            if (label != null && isCommonSupplier) {
                context.write(new Text("Community " + label), new IntWritable(1));
            }
        }
    }

    // ========== Job 4: Count Unique Members per Community ==========
    
    public static class CountMapper extends Mapper<Object, Text, Text, IntWritable> {
        @Override
        public void map(Object key, Text value, Context context) 
                throws IOException, InterruptedException {
            String line = value.toString().trim();
            if (line.isEmpty()) return;
            
            String[] parts = line.split("\\t");
            if (parts.length >= 2) {
                context.write(new Text(parts[0]), new IntWritable(Integer.parseInt(parts[1])));
            }
        }
    }
    
    public static class CountReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context) 
                throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            context.write(key, new IntWritable(sum));
        }
    }

    // ========== Main Driver ==========
    
    public static void main(String[] args) throws Exception {
        if (args.length != 3) {
            System.err.println("Usage: TaskC <relation input> <label input> <output path>");
            System.exit(-1);
        }

        Configuration conf = new Configuration();
        
        // Job 1: Build supplier lists
        Job job1 = Job.getInstance(conf, "Build Supplier Lists");
        job1.setJarByClass(TaskC.class);
        job1.setMapperClass(SupplierListMapper.class);
        job1.setReducerClass(SupplierListReducer.class);
        job1.setOutputKeyClass(Text.class);
        job1.setOutputValueClass(Text.class);
        
        Path job1Input = new Path(args[0]);
        Path job1Output = new Path(args[2] + "_job1");
        TextInputFormat.addInputPath(job1, job1Input);
        FileOutputFormat.setOutputPath(job1, job1Output);
        
        if (!job1.waitForCompletion(true)) {
            System.exit(1);
        }
        
        // Job 2: Find common suppliers
        Job job2 = Job.getInstance(conf, "Find Common Suppliers");
        job2.setJarByClass(TaskC.class);
        job2.setMapperClass(CommonSupplierMapper.class);
        job2.setReducerClass(CommonSupplierReducer.class);
        job2.setOutputKeyClass(Text.class);
        job2.setOutputValueClass(Text.class);
        job2.setNumReduceTasks(1);
        
        Path job2Output = new Path(args[2] + "_job2");
        TextInputFormat.addInputPath(job2, job1Output);
        FileOutputFormat.setOutputPath(job2, job2Output);
        
        if (!job2.waitForCompletion(true)) {
            System.exit(1);
        }
        
        // Job 3: Join with labels
        Job job3 = Job.getInstance(conf, "Join with Labels");
        job3.setJarByClass(TaskC.class);
        
        MultipleInputs.addInputPath(job3, new Path(args[1]), TextInputFormat.class, LabelMapper.class);
        MultipleInputs.addInputPath(job3, job2Output, TextInputFormat.class, SupplierMapper.class);
        
        job3.setReducerClass(JoinReducer.class);
        job3.setOutputKeyClass(Text.class);
        job3.setOutputValueClass(IntWritable.class);
        
        Path job3Output = new Path(args[2] + "_job3");
        FileOutputFormat.setOutputPath(job3, job3Output);
        
        if (!job3.waitForCompletion(true)) {
            System.exit(1);
        }
        
        // Job 4: Count by community
        Job job4 = Job.getInstance(conf, "Count by Community");
        job4.setJarByClass(TaskC.class);
        job4.setMapperClass(CountMapper.class);
        job4.setReducerClass(CountReducer.class);
        job4.setOutputKeyClass(Text.class);
        job4.setOutputValueClass(IntWritable.class);
        
        TextInputFormat.addInputPath(job4, job3Output);
        FileOutputFormat.setOutputPath(job4, new Path(args[2]));
        
        System.exit(job4.waitForCompletion(true) ? 0 : 1);
    }
}
