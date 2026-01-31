package edu.cuhk.iems5730;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.*;

/**
 * Task E: Find the TOP K (K=4) most similar companies for the large dataset.
 * 
 * Uses composite key and secondary sorting to handle large dataset efficiently.
 * This approach avoids memory exhaustion by using Hadoop's sorting capabilities.
 * 
 * Jobs:
 * Job1: Build supplier lists
 * Job2: Calculate similarities with composite key for efficient top-K selection
 */
public class TaskE {

    // ========== Composite Key for Secondary Sorting ==========
    
    public static class CompanySimilarityKey implements WritableComparable<CompanySimilarityKey> {
        private String companyId;
        private double similarity;
        private String otherCompanyId;
        
        public CompanySimilarityKey() {}
        
        public CompanySimilarityKey(String companyId, double similarity, String otherCompanyId) {
            this.companyId = companyId;
            this.similarity = similarity;
            this.otherCompanyId = otherCompanyId;
        }
        
        @Override
        public void write(DataOutput out) throws IOException {
            out.writeUTF(companyId);
            out.writeDouble(similarity);
            out.writeUTF(otherCompanyId);
        }
        
        @Override
        public void readFields(DataInput in) throws IOException {
            companyId = in.readUTF();
            similarity = in.readDouble();
            otherCompanyId = in.readUTF();
        }
        
        @Override
        public int compareTo(CompanySimilarityKey other) {
            // First, group by company ID
            int cmp = this.companyId.compareTo(other.companyId);
            if (cmp != 0) return cmp;
            
            // Then sort by similarity (descending)
            cmp = Double.compare(other.similarity, this.similarity);
            if (cmp != 0) return cmp;
            
            // Finally, sort by other company ID (ascending) for tie-breaking
            return this.otherCompanyId.compareTo(other.otherCompanyId);
        }
        
        public String getCompanyId() { return companyId; }
        public double getSimilarity() { return similarity; }
        public String getOtherCompanyId() { return otherCompanyId; }
    }
    
    // ========== Natural Key Grouping Comparator ==========
    
    public static class NaturalKeyGroupingComparator extends WritableComparator {
        protected NaturalKeyGroupingComparator() {
            super(CompanySimilarityKey.class, true);
        }
        
        @Override
        public int compare(WritableComparable w1, WritableComparable w2) {
            CompanySimilarityKey k1 = (CompanySimilarityKey) w1;
            CompanySimilarityKey k2 = (CompanySimilarityKey) w2;
            return k1.getCompanyId().compareTo(k2.getCompanyId());
        }
    }
    
    // ========== Partitioner ==========
    
    public static class CompanyPartitioner extends Partitioner<CompanySimilarityKey, Text> {
        @Override
        public int getPartition(CompanySimilarityKey key, Text value, int numPartitions) {
            return (key.getCompanyId().hashCode() & Integer.MAX_VALUE) % numPartitions;
        }
    }

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

    // ========== Job 2: Calculate Similarities with Composite Key ==========
    
    public static class SimilarityMapper extends Mapper<Object, Text, Text, Text> {
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

    public static class SimilarityReducer extends Reducer<Text, Text, CompanySimilarityKey, Text> {
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
                        
                        // Use composite key for sorting
                        context.write(
                            new CompanySimilarityKey(c1.companyId, similarity, c2.companyId),
                            new Text(commonStr)
                        );
                        
                        context.write(
                            new CompanySimilarityKey(c2.companyId, similarity, c1.companyId),
                            new Text(commonStr)
                        );
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

    // ========== Job 3: Select Top K with Secondary Sorting ==========
    
    public static class TopKMapper extends Mapper<Object, Text, CompanySimilarityKey, Text> {
        @Override
        public void map(Object key, Text value, Context context) 
                throws IOException, InterruptedException {
            String line = value.toString().trim();
            if (line.isEmpty()) return;
            
            // Parse the composite key output from previous job
            // Format: companyId similarity otherCompanyId\tcommonSuppliers
            String[] parts = line.split("\\s+");
            if (parts.length >= 3) {
                try {
                    String companyId = parts[0];
                    double similarity = Double.parseDouble(parts[1]);
                    String otherCompanyId = parts[2];
                    String commonSuppliers = parts.length > 3 ? parts[3] : "{}";
                    
                    CompanySimilarityKey compositeKey = new CompanySimilarityKey(
                        companyId, similarity, otherCompanyId
                    );
                    context.write(compositeKey, new Text(commonSuppliers));
                } catch (NumberFormatException e) {
                    // Skip malformed lines
                }
            }
        }
    }
    
    public static class TopKReducer extends Reducer<CompanySimilarityKey, Text, Text, Text> {
        private int K = 4;
        private String currentCompany = null;
        private int count = 0;
        
        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            K = context.getConfiguration().getInt("topk.k", 4);
        }
        
        @Override
        public void reduce(CompanySimilarityKey key, Iterable<Text> values, Context context) 
                throws IOException, InterruptedException {
            
            // Reset counter when we encounter a new company
            if (currentCompany == null || !currentCompany.equals(key.getCompanyId())) {
                currentCompany = key.getCompanyId();
                count = 0;
            }
            
            // Only output top K
            if (count < K) {
                for (Text val : values) {
                    String output = key.getCompanyId() + ":" + key.getOtherCompanyId() + 
                                   ", " + val.toString() + ", " + 
                                   String.format("%.6f", key.getSimilarity());
                    context.write(new Text(output), new Text(""));
                    count++;
                    if (count >= K) break;
                }
            }
        }
    }

    // ========== Main Driver ==========
    
    public static void main(String[] args) throws Exception {
        if (args.length < 2 || args.length > 3) {
            System.err.println("Usage: TaskE <input path> <output path> [K]");
            System.exit(-1);
        }

        Configuration conf = new Configuration();
        int K = 4;
        if (args.length == 3) {
            K = Integer.parseInt(args[2]);
        }
        conf.setInt("topk.k", K);
        
        // Enable compression for large dataset
        conf.setBoolean("mapreduce.map.output.compress", true);
        conf.set("mapreduce.map.output.compress.codec", "org.apache.hadoop.io.compress.SnappyCodec");
        
        // Job 1: Build supplier lists
        Job job1 = Job.getInstance(conf, "Build Supplier Lists");
        job1.setJarByClass(TaskE.class);
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
        
        // Job 2: Calculate similarity with composite keys
        Job job2 = Job.getInstance(conf, "Calculate Similarities");
        job2.setJarByClass(TaskE.class);
        job2.setMapperClass(SimilarityMapper.class);
        job2.setReducerClass(SimilarityReducer.class);
        job2.setMapOutputKeyClass(Text.class);
        job2.setMapOutputValueClass(Text.class);
        job2.setOutputKeyClass(CompanySimilarityKey.class);
        job2.setOutputValueClass(Text.class);
        job2.setNumReduceTasks(1);
        
        FileInputFormat.addInputPath(job2, job1Output);
        Path job2Output = new Path(args[1] + "_job2");
        FileOutputFormat.setOutputPath(job2, job2Output);
        
        if (!job2.waitForCompletion(true)) {
            System.exit(1);
        }
        
        // Job 3: Select top K with secondary sorting
        Job job3 = Job.getInstance(conf, "Select Top K with Secondary Sorting");
        job3.setJarByClass(TaskE.class);
        job3.setMapperClass(TopKMapper.class);
        job3.setReducerClass(TopKReducer.class);
        
        job3.setMapOutputKeyClass(CompanySimilarityKey.class);
        job3.setMapOutputValueClass(Text.class);
        job3.setOutputKeyClass(Text.class);
        job3.setOutputValueClass(Text.class);
        
        // Set partitioner and grouping comparator for secondary sorting
        job3.setPartitionerClass(CompanyPartitioner.class);
        job3.setGroupingComparatorClass(NaturalKeyGroupingComparator.class);
        
        FileInputFormat.addInputPath(job3, job2Output);
        FileOutputFormat.setOutputPath(job3, new Path(args[1]));
        
        System.exit(job3.waitForCompletion(true) ? 0 : 1);
    }
}
