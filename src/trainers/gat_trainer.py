from trainers.general_trainer import * 
from torch_geometric.loader import NeighborLoader
from collections import OrderedDict

import json
import sys
import csv
import matplotlib.pyplot as plt 
sys.path.append("..")
from data import LoadData
from models.gat import GAT
from models.hetero_gat import GATSep

def train_gat_on_dataset(
    run_config,
    dataset: utils.Dataset,
    device,
    print_graphs=False,
    seeds=[1],
    test_dataset=None,
    feed_hidden_layer=False,
    sample_neighbors=False,
):

    print(
        "Training gat dataset={}, test_dataset={}".format(
            dataset, test_dataset
        )
    )
    
    all_outputs = []
    test_losses = []
    test_accuracies = []
    val_losses = []
    val_accuracies = []
    f1_scores = []
    rare_f1_scores = []
    best_epochs = []
    is_rare = "twitch" in dataset.value

    run_config.hidden_size = run_config.hidden_size//run_config.attn_heads

    for i in range(len(seeds)):
        set_torch_seed(seeds[i])
        rng = np.random.default_rng(seeds[i])
        
        print("We run for seed {}".format(seeds[i]))
        data_loader = LoadData(
            dataset,
            rng=rng,
            rng_seed=seeds[i],
            test_dataset=test_dataset,
            split_num_for_dataset = i%10,
            inductive = run_config.inductive
        )
        
        data_loader.train_data = data_loader.train_data.to(device, 'x', 'y')
        num_classes = data_loader.num_classes
        train_features = data_loader.train_features
        train_labels = data_loader.train_labels
        num_nodes = len(train_labels)
        
        num_train_nodes = (data_loader.train_mask == True).sum().item()
        batch_size = run_config.batch_size if run_config.batch_size else num_train_nodes
        num_neighbors = [25, 25] if sample_neighbors else [-1, -1]

        train_loader = NeighborLoader(
            data_loader.train_data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=data_loader.train_mask,
            **{'shuffle': True}
        )

        data_loader.val_data = data_loader.val_data.to(device, 'x', 'y')
        val_loader = NeighborLoader(
            data_loader.val_data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=data_loader.val_mask,
            **{'shuffle': True}
        )

        if num_classes > 2:
            is_rare = False

        if data_loader.test_dataset:
            test_dataset = data_loader.test_dataset

        if run_config.hetero:
            model = GATSep(input_size=train_features.size(1),
                        hidden_size=run_config.hidden_size,
                        output_size=num_classes,
                        dropout=run_config.dropout,
                        alpha=0.2,
                        num_hidden=run_config.num_hidden,
                        num_heads=run_config.attn_heads,
                        feed_hidden_layer=feed_hidden_layer)
        else:
            model = GAT(input_size=train_features.size(1),
                        hidden_size=run_config.hidden_size,
                        output_size=num_classes,
                        dropout=run_config.dropout,
                        alpha=0.2,
                        num_hidden=run_config.num_hidden,
                        num_heads=run_config.attn_heads,
                        feed_hidden_layer=feed_hidden_layer)            

        if run_config.parse_communication_results:
            layers_size, _ = model.get_layers_size()
            embeddings_size = model.get_embeddings_size()
            gradients_size = model.get_gradients_size()
            
            comm_size = {}
            comm_size_with_server = {}

            with open(f"comm_results/{dataset}_{model.model_name}.json") as f:
                comm_data = json.load(f)
                
                for node in comm_data:
                    l_size = 0
                    for i in range(len(layers_size)):
                        l_size += (comm_data[node]["layers"][i] * layers_size[i])
                        
                    e_size = 0
                    for i in range(len(embeddings_size)):
                        e_size += (comm_data[node]["embeddings"][i] * embeddings_size[i])
                    
                    g_size = 0
                    for i in range(len(gradients_size)):
                        g_size += (comm_data[node]["gradients"][i] * gradients_size[i])
                    
                    comm_size[node] = l_size + e_size + g_size
            
            with open(f"comm_results/{dataset}_{model.model_name}_with_server.json") as f:
                comm_data_with_server = json.load(f)
                
                for node in comm_data_with_server:
                    l_size = 0
                    for i in range(len(layers_size)):
                        l_size += (comm_data_with_server[node]["layers"][i] * layers_size[i])
                        
                    comm_size_with_server[node] = l_size
                    
            with open(f"comm_results/{dataset}_size_mmlp_gat_nl_2_num_heads_8.json") as f:          
                comm_size_mmlp = json.load(f)
                
            with open(f"comm_results/{dataset}_size_mmlp_gat_nl_2_num_heads_8_with_server.json") as f:
                comm_size_with_server_mmlp = json.load(f)
                
            for node in comm_size:
                comm_size[node] += comm_size_with_server[node]
                comm_size_mmlp[node] += comm_size_with_server_mmlp[node]
            
            sorted_comm_size = sorted(comm_size.items(), key=lambda x:x[1], reverse=True)
            sorted_comm_size = dict(sorted_comm_size)
            
            sorted_comm_size_mmlp = OrderedDict((k, comm_size_mmlp[k]) for k in list(sorted_comm_size.keys()))
            with open(f"comm_results/{dataset}_sorted_size_gat.json", "w") as f:
                json.dump(sorted_comm_size, f, indent = 4)
            
            with open(f"comm_results/{dataset}_sorted_size_mmlp_gat.json", "w") as f:
                json.dump(sorted_comm_size_mmlp, f, indent=4) 
                                                
            plt.plot(list(sorted_comm_size.keys()), list(sorted_comm_size.values()), color='red')
            plt.plot(list(sorted_comm_size_mmlp.keys()), list(sorted_comm_size_mmlp.values()), color='blue')
            plt.savefig(f"comm_results/comm_size_{model.model_name}_{dataset}")
            
            f = csv.writer(open(f'comm_results/plot_values_gat_{dataset}.csv', 'w', encoding='utf8'))
            f.writerow(['Sno', 'node'])
            
            for a, node in enumerate(sorted_comm_size):
                f.writerow([a, sorted_comm_size[node]])
                
            p = csv.writer(open(f'comm_results/plot_values_mmlp_gat_{dataset}.csv', 'w', encoding='utf8'))
            p.writerow(['Sno', 'node'])
            
            for a, node in enumerate(sorted_comm_size_mmlp):
                p.writerow([a, sorted_comm_size_mmlp[node]])
              
            return
                
        trainer = Trainer(model, rng, seed=seeds[i])
        kwargs = {}
        
        if run_config.calculate_communication:
            comm_stats = {}
            comm_stats_with_server = {}
        
            for x in range(num_nodes):
                layers = np.zeros(len(model.convs))
                layers_server = np.zeros(len(model.convs))
                embeddings = np.zeros(len(model.convs) + 1)
                gradients = np.zeros(len(model.convs))
                nei_embeddings = [set() for _ in range(len(model.convs) + 1)]
                            
                comm_stats[x] =  {
                    "layers": layers,
                    "embeddings": embeddings,
                    "gradients": gradients,
                    "nei_embeddings": nei_embeddings,
                }
                
                comm_stats_with_server[x] = {
                    "layers": layers_server,
                }                    
        
            kwargs["comm_stats"] = comm_stats
            kwargs["comm_stats_with_server"] = comm_stats_with_server
            kwargs["is_mmlp"] = False
            kwargs["train_model"] = -1
            

        val_loss, val_acc, best_epoch = trainer.train(
            dataset,
            train_loader,
            val_loader,
            device,
            run_config,
            test_dataset=test_dataset,
            sample_neighbors=sample_neighbors,
            feed_hidden_layer=feed_hidden_layer,
            kwargs=kwargs
        )
        
        data_loader.test_data = data_loader.test_data.to(device, 'x', 'y')
        test_loader = NeighborLoader(
            data_loader.test_data,
            num_neighbors=num_neighbors,
            batch_size=data_loader.test_mask.sum().item(),
            input_nodes=data_loader.test_mask,
            **{'shuffle': True}
        )

        if run_config.calculate_communication:
            comm_result = {}
            comm_result_with_server = {}
            
            for node in kwargs["comm_stats"]:
                comm_result[node] = {
                    "layers": kwargs["comm_stats"][node]["layers"].tolist(),
                    "embeddings": kwargs["comm_stats"][node]["embeddings"].tolist(),
                    "gradients": kwargs["comm_stats"][node]["gradients"].tolist()
                }
                
                comm_result_with_server[node] = {
                    "layers": kwargs["comm_stats_with_server"][node]["layers"].tolist()
                }
                
        test_loss, test_acc, f1_score, rare_f1_score, out_labels, logits = trainer.evaluate(
            test_loader, run_config, is_rare=is_rare, kwargs=kwargs
        )
        
        if run_config.calculate_communication:
            with open(f"comm_results/{dataset}_{model.model_name}.json", "w") as f:
                json.dump(comm_result, f, indent = 4, sort_keys=True)
                
            with open(f"comm_results/{dataset}_{model.model_name}_with_server.json", "w") as f:
                json.dump(comm_result_with_server, f, indent = 4, sort_keys=True)

        all_outputs.append((out_labels, logits))
        test_accuracies.append(test_acc)
        test_losses.append(test_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        f1_scores.append(f1_score)
        best_epochs.append(best_epoch)   
        
        if is_rare:
            rare_f1_score.append(rare_f1_score)             
            
    train_stats = TrainStats(
        run_config,
        dataset,
        model.model_name,
        all_outputs,
        val_losses,
        val_accuracies,
        test_losses,
        test_accuracies,
        best_epochs,
        seeds,
        f1_scores,
        rare_f1_scores,
        test_dataset,
    )
    
    train_stats.print_stats()

    return train_stats   