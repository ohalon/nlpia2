Namespace(annealing_loss_improvement_pct=1.0, batch_size=20, bptt=35, clip=0.25, cuda=True, data='./data/wikitext-2', device='', dropout=0.2, dry_run=False, emsize=200, epochs=10, log_interval=200, lr=20, model='RNN_TANH', nhead=2, nhid=200, nlayers=1, onnx_export='', save='model_epochs_10_model_RNN_TANH_nhid_200_batch_size_20_bptt_35_nlayers_1.pt', seed=1111, tied=False)
| end of epoch   6 | time: 23.78s | valid loss  4.94 | valid ppl   140.41
 | End of training | test loss  4.87 | test ppl   130.37
=========================================================================================
python main.py --cuda --epochs 6 --model GRU --nhid 200 --batch_size 20 --bptt 35 --nlayers 1 --save model_GRU_nlayers_1_epochs_6_nhid_200_batch_size_20_bptt_35_nlayers.pt

 notes/model_GRU_nlayers_2_epochs_12_nhid_200_batch_size_20_bptt_35_nlayers.md <==
| end of epoch  12 | time: 36.28s | valid loss  5.21 | valid ppl   183.12
 | End of training | test loss  5.13 | test ppl   168.89
=========================================================================================
python main.py --cuda --epochs 12 --model GRU --nhid 200 --batch_size 20 --bptt 35 --nlayers 2 --save model_GRU_nlayers_2_epochs_12_nhid_200_batch_size_20_bptt_35_nlayers.pt

 notes/model_LSTM_nlayers_1_epochs_6_nhid_200_batch_size_20_bptt_35_nlayers.md <==
| end of epoch   6 | time: 23.52s | valid loss  5.03 | valid ppl   152.69
 | End of training | test loss  4.96 | test ppl   142.69
=========================================================================================
python main.py --cuda --epochs 6 --model LSTM --nhid 200 --batch_size 20 --bptt 35 --nlayers 1 --save model_LSTM_nlayers_1_epochs_6_nhid_200_batch_size_20_bptt_35_nlayers.pt

 notes/model_LSTM_nlayers_2_epochs_12_nhid_200_batch_size_20_bptt_35_nlayers.md <==
| end of epoch  12 | time: 36.78s | valid loss  4.81 | valid ppl   123.27
 | End of training | test loss  4.74 | test ppl   114.95
=========================================================================================
python main.py --cuda --epochs 12 --model LSTM --nhid 200 --batch_size 20 --bptt 35 --nlayers 2 --save model_LSTM_nlayers_2_epochs_12_nhid_200_batch_size_20_bptt_35_nlayers.pt

 notes/model_RNN_RELU_epochs_5.md <==
 end of epoch   5 | time: 35.32s | valid loss   nan | valid ppl      nan
 | End of training | test loss   nan | test ppl      nan
 notes/model_RNN_RELU_nlayers_1_epochs_6_nhid_200_batch_size_20_bptt_35_nlayers.md <==
| end of epoch   6 | time: 23.05s | valid loss   nan | valid ppl      nan
 | End of training | test loss   nan | test ppl      nan
=========================================================================================
python main.py --cuda --epochs 6 --model RNN_RELU --nhid 200 --batch_size 20 --bptt 35 --nlayers 1 --save model_RNN_RELU_nlayers_1_epochs_6_nhid_200_batch_size_20_bptt_35_nlayers.pt

 notes/model_RNN_RELU_nlayers_2_epochs_12_nhid_200_batch_size_20_bptt_35_nlayers.md <==
| end of epoch  12 | time: 35.32s | valid loss   nan | valid ppl      nan
 | End of training | test loss   nan | test ppl      nan
=========================================================================================
python main.py --cuda --epochs 12 --model RNN_RELU --nhid 200 --batch_size 20 --bptt 35 --nlayers 2 --save model_RNN_RELU_nlayers_2_epochs_12_nhid_200_batch_size_20_bptt_35_nlayers.pt

 notes/model_RNN_TANH_nlayers_1_epochs_6_nhid_200_batch_size_20_bptt_35_nlayers.md <==
| end of epoch   6 | time: 22.94s | valid loss  7.89 | valid ppl  2661.77
 | End of training | test loss  7.70 | test ppl  2205.54
=========================================================================================
python main.py --cuda --epochs 6 --model RNN_TANH --nhid 200 --batch_size 20 --bptt 35 --nlayers 1 --save model_RNN_TANH_nlayers_1_epochs_6_nhid_200_batch_size_20_bptt_35_nlayers.pt

 notes/model_RNN_TANH_nlayers_2_epochs_12_nhid_200_batch_size_20_bptt_35_nlayers.md <==
| end of epoch  12 | time: 35.26s | valid loss  6.91 | valid ppl  1005.60
 | End of training | test loss  6.85 | test ppl   944.04
=========================================================================================
python main.py --cuda --epochs 12 --model RNN_TANH --nhid 200 --batch_size 20 --bptt 35 --nlayers 2 --save model_RNN_TANH_nlayers_2_epochs_12_nhid_200_batch_size_20_bptt_35_nlayers.pt
