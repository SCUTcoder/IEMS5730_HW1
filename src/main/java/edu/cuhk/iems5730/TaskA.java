package edu.cuhk.iems5730;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.IOException;
import java.util.*;

/**
 * Task A: For EVERY company, recommend the company with the maximal number of common suppliers.
 * 
 * This requires 3 MapReduce jobs:
 * Job1: Extract all supplier relationships (buyer -> list of suppliers)
 * Job2: Find all common suppliers between company pairs
 * Job3: Select the company with max common suppliers for each company
 */
public class TaskA {

    // ========== Job 1: Build Supplier List for Each Company ==========
    
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
            
            // Output: company -> supplier1,supplier2,supplier3,...
            if (!suppliers.isEmpty()) {
                context.write(key, new Text(String.join(",", suppliers)));
            }
        }
    }

    // ========== Job 2: Find Common Suppliers Between All Company Pairs ==========
    
    public static class CommonSupplierMapper extends Mapper<Object, Text, Text, Text> {
        private int numReducers = 32; // Will be set from job config
        
        @Override
        protected void setup(Context context) {
            numReducers = context.getNumReduceTasks();
        }
        
        @Override
        public void map(Object key, Text value, Context context) 
                throws IOException, InterruptedException {
            String line = value.toString().trim();
            if (line.isEmpty()) return;
            
            String[] parts = line.split("\\t");
            if (parts.length >= 2) {
                String company = parts[0];
                String supplierList = parts[1];
                
                // Send this company to ALL reducer partitions
                // Each partition will compare companies it receives
                for (int i = 0; i < numReducers; i++) {
                    context.write(new Text(String.valueOf(i)), 
                                 new Text(company + ":" + supplierList));
                }
            }
        }
    }

    public static class CommonSupplierReducer extends Reducer<Text, Text, Text, Text> {
        @Override
        public void reduce(Text key, Iterable<Text> values, Context context) 
                throws IOException, InterruptedException {
            
            // Collect all companies in this partition
            List<CompanySuppliers> companies = new ArrayList<>();
            for (Text val : values) {
                String[] parts = val.toString().split(":", 2);
                if (parts.length == 2) {
                    String companyId = parts[0];
                    Set<String> suppliers = new HashSet<>(Arrays.asList(parts[1].split(",")));
                    companies.add(new CompanySuppliers(companyId, suppliers));
                }
            }
            
            int partitionId = Integer.parseInt(key.toString());
            int totalPartitions = context.getNumReduceTasks();
            
            // Compare all pairs in this partition
            // Use partition ID to distribute work evenly
            for (int i = 0; i < companies.size(); i++) {
                for (int j = i + 1; j < companies.size(); j++) {
                    CompanySuppliers c1 = companies.get(i);
                    CompanySuppliers c2 = companies.get(j);
                    
                    // Determine which partition should handle this pair
                    // Use hash of sorted company IDs
                    String pairKey = c1.companyId.compareTo(c2.companyId) < 0
                        ? c1.companyId + ":" + c2.companyId
                        : c2.companyId + ":" + c1.companyId;
                    int assignedPartition = Math.abs(pairKey.hashCode()) % totalPartitions;
                    
                    // Only process if this partition is responsible
                    if (assignedPartition == partitionId) {
                        Set<String> common = new HashSet<>(c1.suppliers);
                        common.retainAll(c2.suppliers);
                        
                        if (!common.isEmpty()) {
                            List<String> commonList = new ArrayList<>(common);
                            Collections.sort(commonList);
                            String commonStr = "{" + String.join(",", commonList) + "}";
                            
                            // Output for both companies
                            context.write(new Text(c1.companyId), 
                                         new Text(c2.companyId + "," + commonStr + "," + common.size()));
                            context.write(new Text(c2.companyId), 
                                         new Text(c1.companyId + "," + commonStr + "," + common.size()));
                        }
                    }
                }
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

    // ========== Job 3: Select Max Common Suppliers for Each Company ==========
    
    public static class MaxCommonMapper extends Mapper<Object, Text, Text, Text> {
        @Override
        public void map(Object key, Text value, Context context) 
                throws IOException, InterruptedException {
            String line = value.toString().trim();
            if (line.isEmpty()) return;
            
            String[] parts = line.split("\\t");
            if (parts.length >= 2) {
                context.write(new Text(parts[0]), new Text(parts[1]));
            }
        }
    }

    public static class MaxCommonReducer extends Reducer<Text, Text, Text, Text> {
        @Override
        public void reduce(Text key, Iterable<Text> values, Context context) 
                throws IOException, InterruptedException {
            
            String bestCompany = null;
            String bestCommonSuppliers = null;
            int maxCount = 0;
            
            for (Text val : values) {
                String[] parts = val.toString().split(",", 3);
                if (parts.length == 3) {
                    String otherCompany = parts[0];
                    String commonSuppliers = parts[1];
                    int count = Integer.parseInt(parts[2]);
                    
                    // Keep the one with max count, or smaller ID if tied
                    if (count > maxCount || 
                        (count == maxCount && (bestCompany == null || otherCompany.compareTo(bestCompany) < 0))) {
                        maxCount = count;
                        bestCompany = otherCompany;
                        bestCommonSuppliers = commonSuppliers;
                    }
                }
            }
            
            if (bestCompany != null) {
                // Output format: A:B, {C,E}, 2
                String output = key.toString() + ":" + bestCompany + ", " + 
                               bestCommonSuppliers + ", " + maxCount;
                context.write(new Text(output), new Text(""));
            }
        }
    }

    // ========== Main Driver ==========
    
    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("Usage: TaskA <input path> <output path>");
            System.exit(-1);
        }

        Configuration conf = new Configuration();
        
        // Job 1: Build supplier lists
        Job job1 = Job.getInstance(conf, "Build Supplier Lists");
        job1.setJarByClass(TaskA.class);
        job1.setMapperClass(SupplierListMapper.class);
        job1.setReducerClass(SupplierListReducer.class);
        job1.setOutputKeyClass(Text.class);
        job1.setOutputValueClass(Text.class);
        job1.setNumReduceTasks(8);
        
        FileInputFormat.addInputPath(job1, new Path(args[0]));
        Path job1Output = new Path(args[1] + "_job1");
        FileOutputFormat.setOutputPath(job1, job1Output);
        
        if (!job1.waitForCompletion(true)) {
            System.exit(1);
        }
        
        // Job 2: Find common suppliers
        Job job2 = Job.getInstance(conf, "Find Common Suppliers");
        job2.setJarByClass(TaskA.class);
        job2.setMapperClass(CommonSupplierMapper.class);
        job2.setReducerClass(CommonSupplierReducer.class);
        job2.setOutputKeyClass(Text.class);
        job2.setOutputValueClass(Text.class);
        job2.setNumReduceTasks(32); // Multiple reducers to distribute pair comparison
        
        FileInputFormat.addInputPath(job2, job1Output);
        Path job2Output = new Path(args[1] + "_job2");
        FileOutputFormat.setOutputPath(job2, job2Output);
        
        if (!job2.waitForCompletion(true)) {
            System.exit(1);
        }
        
        // Job 3: Select max common suppliers
        Job job3 = Job.getInstance(conf, "Select Max Common Suppliers");
        job3.setJarByClass(TaskA.class);
        job3.setMapperClass(MaxCommonMapper.class);
        job3.setReducerClass(MaxCommonReducer.class);
        job3.setOutputKeyClass(Text.class);
        job3.setOutputValueClass(Text.class);
        job3.setNumReduceTasks(8); // Multiple reducers for parallel processing
        
        FileInputFormat.addInputPath(job3, job2Output);
        FileOutputFormat.setOutputPath(job3, new Path(args[1]));
        
        System.exit(job3.waitForCompletion(true) ? 0 : 1);
    }
}
