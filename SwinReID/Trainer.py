from tqdm import tqdm
import torch
import os
from einops import rearrange
import pandas as pd
class ReIDTrainer:
    '''
        A wrapper class that launches the training process 
    '''
    
    def __init__(self, net, backbone,ce_loss_fn=None, triplet_loss_fn=None,center_loss_fn=None, optimizer=None, optimizer_center=None,lr_scheduler=None, device='cpu'):
        '''
            Args: 
                net (nn.Module): the network to be trained 
                ce_loss_fn (CrossEntropyLoss): cross entropy loss function from Pytorch
                triplet_loss_fn (TripletMarginLoss): triplet loss function from Pytorch
                optimizer (torch.optim): optimizer for `net`
                lr_scheduler (torch.optim.lr_scheduler): scheduler for `optimizer`
                device (str, 'cuda' or 'cpu'): the device to train the model 
        '''
        self.device = device
        self.net = net.to(self.device)

        print(f'Training on {self.device}')

        self.ce_loss_fn = ce_loss_fn
        self.triplet_loss_fn = triplet_loss_fn
        self.center_loss_fn = center_loss_fn

        self.optimizer = optimizer
        self.optimizer_center = optimizer_center
        self.scheduler = lr_scheduler
        self.backbone = backbone

        self.metrics = []

    def fit(self, train_loader, test_loader, epochs, gt_index_path, name_query_path, jk_index_path, save_dir, early_stopping, check_init=False,similarity='cosine'):
        '''
            Train the model for `epochs` epochs, where each epoch is composed of a training step and a testing step similarity

            Args:
                train_loader (DataLoader): a dataloader that wraps the training dataset (a Veri776Train instance)
                epochs (int): epochs
                gt_index_path (str): the path to gt_index.txt under veri776 root folder
                name_query_path (str): the path to name_query.txt under veri776 root folder
                save_dir (str): path to save the model
                check_init (boolean): if true, then test the model with initial weight
                early_stopping (int): if the performance on validation set stop improving for a continuous `early_stopping` epochs, the `fit` method returns control
        '''

        # if the save if provided and the path hasn't existed yet
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)


        name_queries = self.get_queries(name_query_path)
        gt_indices = self.get_label_indices(gt_index_path)
        


        if check_init:
            print('check init')
            self.test(
                test_loader=test_loader, 
                gt_indices=gt_indices, 
                name_queries=name_queries,
                save_dir = save_dir,
                similarity = similarity
            )

        best_val = 0
        patience = 0

        for epoch in range(epochs):
            # self.train(train_loader,epoch,epochs)
            triplet_loss, ce_loss, center_loss = self.train(train_loader, epoch, epochs)
            complete_hits, gt_hits = self.test(
                test_loader=test_loader, 
                gt_indices=gt_indices, 
                name_queries=name_queries,
                save_dir = save_dir,
                similarity=similarity
            )

            if complete_hits + gt_hits > best_val:
                print(f"New best model found!")
                print(f"complete_hits: {complete_hits}, gt_hits: {gt_hits}, total: {complete_hits + gt_hits}")
                print("Saving the model...")  # Add code here to save the model
                # print('save model saved')
                best_val = complete_hits + gt_hits
                patience = 0
                if self.center_loss_fn is not None:
                    state = {
                        'model_state_dict': self.net.state_dict(),
                        'center_state_dict': self.center_loss_fn.state_dict(),  # Save center weights
                    }
                    torch.save(state, os.path.join(save_dir, f'{self.backbone}_centerloss_best.pth'))
                else:
                    torch.save(self.net.state_dict(), os.path.join(save_dir, f'{self.backbone}_best.pth'))
                
                    # Calculate average losses



            if self.scheduler is not None:
                self.scheduler.step()
            # Append metrics to the list
            self.metrics.append({
                'Epoch': epoch + 1,
                'Triplet Loss': triplet_loss,
                'CE Loss': ce_loss,
                'Center Loss': center_loss,
                'GT Hits': gt_hits / len(name_queries),  # Normalize hits
                'Complete Hits': complete_hits / len(name_queries)  # Normalize hits
            })
            patience += 1
            # Save the metrics to CSV after each epoch
            self.save_metrics_to_csv(save_dir)

            if patience >= early_stopping:
                return

    # def save_metrics(self, save_dir, epoch, total_triplet_loss, total_ce_loss, total_center_loss,complete_hits, gt_hits, metric, backbone):
    #     # Prepare the text file where metrics will be saved
    #     metrics_file = os.path.join(save_dir, 'training_metrics.txt')

    #     # Save the metrics into the file
    #     with open(metrics_file, 'a') as f:
    #         f.write(f"Epoch: {epoch + 1}    ")
    #         f.write(f"Backbone: {backbone}  ")
    #         f.write(f"Triplet Loss: {total_triplet_loss:.4f}    ")
    #         f.write(f"CE Loss: {total_ce_loss:.4f}  ")
    #         f.write(f"Center Loss: {total_center_loss:.4f}  ")
    #         f.write(f"R@1 gt_hits: {gt_hits :.3f}, R@1 complete_hits: {complete_hits :.3f}  ")
    #         f.write(f"Metric: {metric}  ")
    #         f.write(f"Total gt_hits: {gt_hits}, Total complete_hits: {complete_hits}\n")
    #         f.write("--------------------------------------------------\n")
    def save_metrics_to_csv(self, save_dir):


        # Create the file name using the backbone
        if self.center_loss_fn is not None:
            file_name = f"{self.backbone}_centerloss_metrics.csv"
        else:
            file_name = f"{self.backbone}_metrics.csv"
        
        file_path = os.path.join(save_dir, file_name)

        # Convert the latest metrics to a DataFrame
        df = pd.DataFrame(self.metrics)

        # If file doesn't exist, write with headers; else, append without headers
        if not os.path.exists(file_path):
            df.to_csv(file_path, index=False)
        else:
            # Append without headers
            df.tail(1).to_csv(file_path, index=False, mode='a', header=False)

        print(f"Metrics saved to {file_path}")

    def train(self, train_loader, current_epoch, total_epochs):
        '''
            Args:
                train_loader (DataLoader): a dataloader that wraps the training dataset (a ReIDDataset instance)
        '''
        self.net.train()

        total_ce_loss = 0
        total_triplet_loss = 0
        total_center_loss = 0
        for images, labels in tqdm(train_loader, dynamic_ncols=True, desc=f"Training on Epoch {current_epoch + 1}/{total_epochs}"):
            images, labels = images.to(self.device), labels.to(self.device)
            
            anchors = images[:, 0, :].squeeze()
            positvies = images[:, 1, :].squeeze()
            negatives = images[:, 2, :].squeeze()

            anchor_embeddings, _, anchor_out = self.net(anchors)
            positive_embeddings, _, positive_out = self.net(positvies)
            negative_embeddings, _, negative_out = self.net(negatives)

            triplet_loss = self.triplet_loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)

            
            preds = rearrange([anchor_out, positive_out, negative_out], 't b e -> (b t) e')
            labels_batch = torch.flatten(labels)

            ce_loss = self.ce_loss_fn(preds, labels_batch)

            
            # cent_loss = self.center_loss_fn(cent_preds, labels_batch)
            if self.center_loss_fn:
                cent_preds = rearrange([anchor_embeddings, positive_embeddings, negative_embeddings], 't b e -> (b t) e')
                cent_loss = self.center_loss_fn(cent_preds, labels_batch)
                self.optimizer_center.zero_grad()
            else:
                cent_loss = torch.tensor(0.0, device=anchor_embeddings.device)  # Ensure cent_loss is a tensor



            self.optimizer.zero_grad()

            loss = triplet_loss + ce_loss + 3.5e-4 *cent_loss
            # loss = triplet_loss + ce_loss + 3.5e-4 * cent_loss
            # loss = triplet_loss + ce_loss + 3.5e-4 * cent_loss
            loss.backward()
            self.optimizer.step()
            if self.center_loss_fn:
                self.optimizer_center.step()

            

            total_ce_loss += ce_loss.item()
            total_triplet_loss += triplet_loss.item()
            total_center_loss += cent_loss.item()



        print(f'Epoch {current_epoch + 1}/{total_epochs} | Triplet Loss: {total_triplet_loss:.4f}, CE Loss: {total_ce_loss:.4f}, Center Loss: {total_center_loss:.4f}')
        return total_triplet_loss, total_ce_loss, total_center_loss

    @staticmethod
    def get_queries(name_queries_path):
        '''
            Args:
                name_query_path (str): the path to name_query.txt under veri776 root folder
            Returns:
                return the file names in name_query.txt
               
        '''
        with open(name_queries_path) as f:
            return [line.rstrip('\n') for line in f.readlines()]


    @staticmethod
    def get_label_indices(index_path):
        '''
            Args:
                gt_index_path (str): the path to gt_index.txt under veri776 root folder
            Returns:
                a list composed of ground truths
        '''
        with open(index_path) as f:
            lines = f.readlines()
            ls = []
            for line in lines:
                # ls.append([int(x) for x in line.rstrip(' \n').split(' ')])
                ls.append([int(x) for x in line.strip().split()])
                

            return ls


    @torch.no_grad()
    def test(self, test_loader, gt_indices, name_queries,save_dir,similarity):
        '''
            Args:
                test_set (Veri776Test): a Veri776Test instance
                gt_indices (list of list): a list of ground truths (see gt_index.txt)
                name_queries (list of str): a list of query file name (see name_query.txt)
        '''

        self.net.eval()
        # File paths to store the data
        # Define the full file paths for the cache directory
        # test_dict_file = os.path.join(save_dir, 'test_dict.pt')
        # query_indices_map_file = os.path.join(save_dir, 'query_indices_map.pt')
        # test_feats_file = os.path.join(save_dir, 'test_feats.pt')  

        # # Check if the test_dict, query_indices_map, and test_feats already exist
        # test_dict = self.check_and_load(test_dict_file)
        # query_indices_map = self.check_and_load(query_indices_map_file)
        # test_feats = self.check_and_load(test_feats_file)

        # if test_dict is None or query_indices_map is None or test_feats is None:
        #     print("Building test_dict, query_indices_map, and test_feats from scratch...")

        #     # Build test_dict
        #     test_dict = self.build_feature_mapping(test_loader)
        #     self.save_data(test_dict, test_dict_file)

        #     # Build query_indices_map
        #     query_indices_map = self.build_query_indices(test_loader.dataset.img_file_names, name_queries)
        #     self.save_data(query_indices_map, query_indices_map_file)

        #     # Collect features and store
        #     test_feats = torch.stack(list(test_dict.values()))
        #     self.save_data(test_feats, test_feats_file)
        # else:
        #     print("Loaded test_dict, query_indices_map, and test_feats from cache.")  
        # 
        #       
        # query_indices_map_file = os.path.join(save_dir, 'query_indices_map.pt')
        # Check if the test_dict, query_indices_map, and test_feats already exist
  
        # query_indices_map = self.check_and_load(query_indices_map_file)
        # if query_indices_map is None:
        #     print("Building  query_indices_map from scratch...")
        #     # Build query_indices_map
        #     query_indices_map = self.build_query_indices(test_loader.dataset.img_file_names, name_queries)
        #     self.save_data(query_indices_map, query_indices_map_file)

        test_dict = self.build_feature_mapping(test_loader)
        query_indices_map = self.build_query_indices(test_loader.dataset.img_file_names, name_queries)
        
        # img_names = test_set.img_file_names
        test_feats = torch.stack(list(test_dict.values()))

        gt_hits = 0
        complete_hits = 0

        for gt, query in tqdm(zip(gt_indices, name_queries), total=len(gt_indices), dynamic_ncols=True, desc='querying gt'):
            query_feat = test_dict[query]
            # dist = self.get_euclidean_dist(query_feat, test_feats)
            # sorted_arg = torch.argsort(dist) + 1 # the indices are 0-indexed, however, the gt and jk are 1-indexed
           
            # dist = self.get_cosine_similarity(query_feat, test_feats)
            # dist = self.calculate_similarity(query_feat, test_feats, metric)
            # Sort in descending order (so that higher cosine similarity comes first)
            # sorted_arg = torch.argsort(dist, descending=True) + 1  # Add 1 for 1-indexing
            # print("dist : " ,dist)

            sorted_arg = self.calculate_similarity(query_feat, test_feats, similarity)
            # print("sorted_arg : ",sorted_arg)
            gt_hits += self.query_gt(sorted_arg, gt, query_indices_map[query])
            complete_hits = 0
            


        # print(f'R@1 gt_hits: {gt_hits / len(name_queries):.3f}, 
        #       R@1 complete_hits: {complete_hits / len(name_queries):.3f} , 
        #       metric for dist: {metric}')
        print(
            f'R@1 gt_hits: {gt_hits / len(name_queries):.3f}, '
            f'similarity for dist: {similarity}'
        )

        return complete_hits, gt_hits

    def save_data(self, data, file_name):
        torch.save(data, file_name)
    
    def load_data(self, file_name):
        return torch.load(file_name)

    def check_and_load(self, file_name):
        # Check if the file already exists
        if os.path.exists(file_name):
            return self.load_data(file_name)
        return None
    



    def query_gt(self, sorted_arg, gt, query_idx):
        '''
            Return 1 if the closest image which (is not in `jk` and is not itself) is in gt, otherwise return  0
        '''

        for idx in sorted_arg:
            if (idx == query_idx):
                continue

            return 1 if idx in gt else 0



    # def build_query_indices(self, img_file_names, name_queries):
    #     img_ptr = 0
    #     query_ptr = 0

    #     indices = dict()

    #     while query_ptr < len(name_queries) and img_ptr < len(img_file_names):
    #         if name_queries[query_ptr] == img_file_names[img_ptr]:
    #             indices[name_queries[query_ptr]] = img_ptr + 1 # the query, ground truth, ..., should be 1-indexed
    #             query_ptr += 1
    #             img_ptr += 1
    #         else:
    #             img_ptr += 1

    #     return indices
    def build_query_indices(self, img_file_names, name_queries):
        # Sort the image file names in lexicographical order
        sorted_img_file_names = sorted(img_file_names)
        
        img_ptr = 0
        query_ptr = 0
        indices = dict()

        while query_ptr < len(name_queries) and img_ptr < len(sorted_img_file_names):
            if name_queries[query_ptr] == sorted_img_file_names[img_ptr]:
                indices[name_queries[query_ptr]] = img_ptr + 1  # 1-indexed
                query_ptr += 1
                img_ptr += 1
            else:
                img_ptr += 1

        return indices


    def get_euclidean_dist(self, query_feats, test_feats):
        ''' calculate the Euclidean distance between the query and the entire test set 

            Args: 
                query_feats (torch.tensor): the embedding feature representation of the query img
                test_feats (torch.tensor): the embedding feature representations of the entire test set

            Returns:
                returns a 1d torch.tensor where entry `i` represents the distance between the query_feats and the `i`th test_feats
        '''
        # print(f"Initial shape of query_feats: {query_feats.shape}")
        # print(f"Initial shape of test_feats: {test_feats.shape}")
        query_feats = rearrange(query_feats, '(b n f) -> b n f', b=1, n=1)
        test_feats = rearrange(test_feats, '(b n) f -> b n f', b=1)
        # print(f"Shape of query_feats after normalization: {query_feats.shape}")
        # print(f"Shape of test_feats after normalization: {test_feats.shape}")

        dist_mat = torch.cdist(query_feats, test_feats).squeeze()
        

        return dist_mat
    
    def get_cosine_similarity(self, query_feats, test_feats):
        # Print the shapes before processing
        # print(f"Initial shape of query_feats: {query_feats.shape}")
        # print(f"Initial shape of test_feats: {test_feats.shape}")
        
        # Add a batch dimension to query_feats to make it (1, 2048)
        query_feats = query_feats.unsqueeze(0)  # Now query_feats is (1, 2048)

        # Normalize the vectors to unit vectors (cosine similarity requires normalization)
        query_feats = torch.nn.functional.normalize(query_feats, p=2, dim=1)
        test_feats = torch.nn.functional.normalize(test_feats, p=2, dim=1)

        # Compute cosine similarity using matrix multiplication
        similarity = torch.mm(query_feats, test_feats.t())  # (1, 2048) * (2048, 11579) -> (1, 11579)

        # Print the shape of the similarity matrix
        # print(f"Shape of similarity matrix: {similarity.shape}")

        return similarity.squeeze()  # Shape will become (11579)
    def calculate_similarity(self, query_feats, test_feats, similarity):
        if similarity == 'euclidean':
            # Calculate Euclidean distance
            dist = self.get_euclidean_dist(query_feats, test_feats)
            sorted_arg = torch.argsort(dist) + 1

        elif similarity == 'cosine':
            # Calculate Cosine similarity
            dist = self.get_cosine_similarity(query_feats, test_feats)
            sorted_arg = torch.argsort(dist, descending=True) + 1
        return sorted_arg

    
    def build_feature_mapping(self, test_loader):
        """
        Args:
            test_loader (DataLoader): A DataLoader instance for the test set.

        Returns:
            A dictionary which maps the file name in the test set to its embedding feature.
        """
        test_dict = dict()

        for img_names, imgs in tqdm(test_loader, dynamic_ncols=True, desc='Trainer.build_feature_mapping'):
            imgs = imgs.to(self.device)  # Move the batch of images to the appropriate device (GPU/CPU)
            
            # Forward pass through the network for the batch
            _, feats, _ = self.net(imgs)  # Assuming self.net can handle batch processing

            # Loop over each image in the batch
            for img_name, feat in zip(img_names, feats):
                test_dict[img_name] = feat.squeeze()

        return test_dict   
        
    # def build_feature_mapping(self, test_set):
    #     '''
    #         Args:
    #             test_set (Veri776Test): a Veri776Test instance

    #         Returns:
    #             a dictionary which maps the file name in the test set to its embedding feature
    #     '''
    #     test_dict = dict()
        
    #     for img_name, img in tqdm(test_set, dynamic_ncols=True, desc='Trainer.build_feature_mapping'):
    #         img = img.to(self.device)
    #         feat, _, _ = self.net(img.unsqueeze(dim=0)) # use f_t for now

    #         test_dict[img_name] = feat.squeeze()


    #     return test_dict
    