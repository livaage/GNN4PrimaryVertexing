# GNN4PrimaryVertexing

To run graph building, first direct graph_building.py to the correct track file. Then run 
'python graph_building.py' 

The built graphs will take up a fair chunk of memory (~40Gb), so consdier limiting it to a few thousand events

Then run 
'python train_gnn.py' 

This runs the model in models/gcn.py 
