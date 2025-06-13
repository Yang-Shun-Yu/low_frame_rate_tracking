import os

def create_gt_index(query_folder, test_folder, output_file):
    # Get all jpg files from each folder and sort them
    query_files = sorted([f for f in os.listdir(query_folder) if f.lower().endswith('.jpg')])
    test_files = sorted([f for f in os.listdir(test_folder) if f.lower().endswith('.jpg')])
    
    # Open the output file for writing
    with open(output_file, 'w') as out_f:
        # Process each query image
        for query in query_files:
            # Extract the prefix from the query file (everything before the first underscore)
            query_prefix = query.split('_')[0]
            
            # Initialize a list to store matching indices (using 1-indexing)
            matching_indices = []
            
            # Loop through test images, with index
            for idx, test_file in enumerate(test_files):
                # Extract prefix of test image
                test_prefix = test_file.split('_')[0]
                # Check if the prefixes match and the filenames are not exactly the same
                if test_prefix == query_prefix and test_file != query:
                    # Append the 1-indexed position (idx+1) to the list
                    matching_indices.append(str(idx + 1))
            
            # Write the matching indices to the output file on one line
            out_f.write(" ".join(matching_indices) + "\n")

if __name__ == "__main__":
    # Replace with the correct folder names if needed
    image_query_path = '/home/eddy/Desktop/VehicleTracking/image_query'
    image_test_path = '/home/eddy/Desktop/VehicleTracking/image_test'
    gt_index_path = '/home/eddy/Desktop/VehicleTracking/gt_index_s.txt'
    create_gt_index(image_query_path, image_test_path, gt_index_path)
