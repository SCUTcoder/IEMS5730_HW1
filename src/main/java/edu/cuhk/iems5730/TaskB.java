package edu.cuhk.iems5730;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.IOException;
import java.util.*;

/**
 * Task B: Find the TOP K (K=3) most similar companies and their common suppliers.
 * 
 * Similarity = |Common Suppliers| / |Union of Suppliers|
 * 
 * This requires 3 MapReduce jobs:
 * Job1: Build supplier list for each company
 * Job2: Calculate similarity scores for all company pairs
 * Job3: Select top K similar companies for each company
 */
public class TaskB {

    // ========== Job 1: Build Supplier List (Reuse from TaskA) ==========
    
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

    // ========== Job 2: Calculate Similarity Scores ==========
    
    public static class SimilarityMapper extends Mapper<Object, Text, Text, Text> {
        @Override
        public void map(Object key, Text value, Context context) 
                throws IOException, InterruptedException {
            String line = value.toString().trim();
            if (line.isEmpty()) return;
            
            String[] parts = line.split("\\t");
            if (parts.length >= 2) {
                String company = parts[0];
                String supplierList = parts[1];
                
                // Emit to a single key for cross-comparison
                context.write(new Text("ALL"), new Text(company + ":" + supplierList));
            }
        }
    }

    public static class SimilarityReducer extends Reducer<Text, Text, Text, Text> {
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
            
            // Calculate similarity for all pairs
            for (int i = 0; i < companies.size(); i++) {
                for (int j = i + 1; j < companies.size(); j++) {
                    CompanySuppliers c1 = companies.get(i);
                    CompanySuppliers c2 = companies.get(j);
                    
                    Set<String> common = new HashSet<>(c1.suppliers);
                    common.retainAll(c2.suppliers);
                    
                    Set<String> union = new HashSet<>(c1.suppliers);
                    union.addAll(c2.suppliers);
                    
                    double similarity = 0.0;
                    if (!union.isEmpty()) {
                        similarity = (double) common.size() / union.size();
                    }
                    
                    if (similarity > 0) {
                        List<String> commonList = new ArrayList<>(common);
                        Collections.sort(commonList);
                        String commonStr = "{" + String.join(",", commonList) + "}";
                        String simStr = String.format("%.6f", similarity);
                        
                        // Output for company 1
                        context.write(new Text(c1.companyId), 
                                     new Text(c2.companyId + "," + commonStr + "," + simStr));
                        
                        // Output for company 2
                        context.write(new Text(c2.companyId), 
                                     new Text(c1.companyId + "," + commonStr + "," + simStr));
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

    // ========== Job 3: Select Top K Similar Companies ==========
    
    public static class TopKMapper extends Mapper<Object, Text, Text, Text> {
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

    public static class TopKReducer extends Reducer<Text, Text, Text, Text> {
        private int K = 3; // Default K=3
        
        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            K = context.getConfiguration().getInt("topk.k", 3);
        }
        
        @Override
        public void reduce(Text key, Iterable<Text> values, Context context) 
                throws IOException, InterruptedException {
            
            PriorityQueue<SimilarityRecord> topK = new PriorityQueue<>(K + 1, 
                new Comparator<SimilarityRecord>() {
                    @Override
                    public int compare(SimilarityRecord a, SimilarityRecord b) {
                        // Min heap: smaller similarity at top
                        int cmp = Double.compare(a.similarity, b.similarity);
                        if (cmp != 0) return cmp;
                        // If tied, prefer larger ID (so smaller ID stays in heap)
                        return b.companyId.compareTo(a.companyId);
                    }
                });
            
            for (Text val : values) {
                String[] parts = val.toString().split(",", 3);
                if (parts.length == 3) {
                    String otherCompany = parts[0];
                    String commonSuppliers = parts[1];
                    double similarity = Double.parseDouble(parts[2]);
                    
                    topK.offer(new SimilarityRecord(otherCompany, commonSuppliers, similarity));
                    if (topK.size() > K) {
                        topK.poll();
                    }
                }
            }
            
            // Sort in descending order by similarity, then by company ID
            List<SimilarityRecord> results = new ArrayList<>(topK);
            Collections.sort(results, new Comparator<SimilarityRecord>() {
                @Override
                public int compare(SimilarityRecord a, SimilarityRecord b) {
                    int cmp = Double.compare(b.similarity, a.similarity);
                    if (cmp != 0) return cmp;
                    return a.companyId.compareTo(b.companyId);
                }
            });
            
            // Output top K records
            for (SimilarityRecord record : results) {
                String output = key.toString() + ":" + record.companyId + ", " + 
                               record.commonSuppliers + ", " + 
                               String.format("%.6f", record.similarity);
                context.write(new Text(output), new Text(""));
            }
        }
        
        private static class SimilarityRecord {
            String companyId;
            String commonSuppliers;
            double similarity;
            
            SimilarityRecord(String id, String common, double sim) {
                this.companyId = id;
                this.commonSuppliers = common;
                this.similarity = sim;
            }
        }
    }

    // ========== Main Driver ==========
    
    public static void main(String[] args) throws Exception {
        if (args.length < 2 || args.length > 3) {
            System.err.println("Usage: TaskB <input path> <output path> [K]");
            System.exit(-1);
        }

        Configuration conf = new Configuration();
        int K = 3;
        if (args.length == 3) {
            K = Integer.parseInt(args[2]);
        }
        conf.setInt("topk.k", K);
        
        // Job 1: Build supplier lists
        Job job1 = Job.getInstance(conf, "Build Supplier Lists");
        job1.setJarByClass(TaskB.class);
        job1.setMapperClass(SupplierListMapper.class);
        job1.setReducerClass(SupplierListReducer.class);
        job1.setOutputKeyClass(Text.class);
        job1.setOutputValueClass(Text.class);
        
        FileInputFormat.addInputPath(job1, new Path(args[0]));
        Path job1Output = new Path(args[1] + "_job1");
        FileOutputFormat.setOutputPath(job1, job1Output);
        
        if (!job1.waitForCompletion(true)) {
            System.exit(1);
        }
        
        // Job 2: Calculate similarity
        Job job2 = Job.getInstance(conf, "Calculate Similarity Scores");
        job2.setJarByClass(TaskB.class);
        job2.setMapperClass(SimilarityMapper.class);
        job2.setReducerClass(SimilarityReducer.class);
        job2.setOutputKeyClass(Text.class);
        job2.setOutputValueClass(Text.class);
        job2.setNumReduceTasks(1);
        
        FileInputFormat.addInputPath(job2, job1Output);
        Path job2Output = new Path(args[1] + "_job2");
        FileOutputFormat.setOutputPath(job2, job2Output);
        
        if (!job2.waitForCompletion(true)) {
            System.exit(1);
        }
        
        // Job 3: Select top K
        Job job3 = Job.getInstance(conf, "Select Top K Similar Companies");
        job3.setJarByClass(TaskB.class);
        job3.setMapperClass(TopKMapper.class);
        job3.setReducerClass(TopKReducer.class);
        job3.setOutputKeyClass(Text.class);
        job3.setOutputValueClass(Text.class);
        
        FileInputFormat.addInputPath(job3, job2Output);
        FileOutputFormat.setOutputPath(job3, new Path(args[1]));
        
        System.exit(job3.waitForCompletion(true) ? 0 : 1);
    }
}
