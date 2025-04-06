# Hyperparameter Tuning

Example training sizes on a PC:
CPU: 125 GB of main RAM, 24 cores
4 GPUs: 12 GB GPU RAM, 80 cores each 

## defaults
```
python main.py --cuda --epochs 6 --batch_size 20 --bptt 35`
| end of epoch   6 | time: 166.00s | valid loss  5.13 | valid ppl   168.22
| End of training | test loss  5.05 | test ppl   155.90
```

## one long batch
```
python main.py --cuda --epochs 6 --batch_size 1 --bptt 700`
| end of epoch   6 | time: 36.54s | valid loss  5.03 | valid ppl  152.33
| End of training | test loss  4.96 | test ppl   142.83
```

## twenty long batches
```
python main.py --cuda --epochs 6 --batch_size 20 --bptt 700
| end of epoch   6 | time: 26.88s | valid loss  5.91 | valid ppl   369.38
| End of training | test loss  5.83 | test ppl   341.69
```

## defaults
```
python main.py --cuda --epochs 6
| end of epoch   6 | time: 36.88s | valid loss  5.01 | valid ppl   149.47
| End of training | test loss  4.93 | test ppl   138.27
```

## GRU long doc
```
python main.py --cuda --epochs 6 --model GRU --batch_size 1 --bptt 700
| end of epoch   6 | time: 166.22s | valid loss  6.15 | valid ppl   469.68
| End of training | test loss  6.05 | test ppl   424.94
```

## extra large doc and 10 hour training (800 epochs)
```
python main.py --cuda --epochs 83 --model GRU --batch_size 1 --bptt 4092
segfault
| end of epoch  83 | time: 165.54s | valid loss  4.76 | valid ppl   116.29
```

## 10 epoch GRU defaults
```
python main.py --device cuda:1 --epochs 10 --model GRU --nhid 400 --batch_size 20 --bptt 35 | tee model_epochs_10_model_GRU_nhid_400_batch_size_20_bptt_35.md
| end of epoch  10 | time: 49.79s | valid loss  5.48 | valid ppl   240.47
| End of training | test loss  5.05 | test ppl   156.71                                                                        
```

## 10 epoch LSTM defaults
```
python main.py --device cuda:1 --epochs 10 --model LSTM --nhid 200 --batch_size 20 --bptt 35 | tee model_epochs_10_model_LSTM_nhid_200_batch_size_20_bptt_35.md
| end of epoch  10 | time: 36.98s | valid loss  4.86 | valid ppl   128.58
| End of training | test loss  4.78 | test ppl   119.65
```


## 10 epoch GRU 300 nhid
```
NHID=300
MODEL=GRU
SAVE=model_epochs_10_model_${MODEL}_nhid_${NHID}_batch_size_20_bptt_35
python main.py --device cuda:1 --epochs 10 --model $MODEL --nhid $NHID --batch_size 20 --bptt 35 \
    --save ${SAVE}.pt| tee ${SAVE}.md
| end of epoch  10 | time: 40.47s | valid loss  5.33 | valid ppl   206.83
| End of training | test loss  5.10 | test ppl   164.42
```

"""
6 --batch_size 20 --bptt 35             | end of epoch   6 | time: 166.00s | valid loss  5.13 | valid ppl   168.22
6 --batch_size 1 --bptt 700             | end of epoch   6 | time: 36.54s | valid loss  5.03 | valid ppl  152.33
6 --batch_size 20 --bptt 700            | end of epoch   6 | time: 26.88s | valid loss  5.91 | valid ppl   369.38
6                                       | end of epoch   6 | time: 36.88s | valid loss  5.01 | valid ppl   149.47
6 --model GRU --batch_size 1 --bptt 700 | end of epoch   6 | time: 166.22s | valid loss  6.15 | valid ppl   469.68
83 --model GRU --batch_size 1 --bptt 4092 | end of epoch  83 | time: 165.54s | valid loss  4.76 | valid ppl   116.29
model_epochs_10_model_LSTM_nhid_200_batch_size_20_bptt_35.md | end of epoch  10 | time: 37.21s | valid loss  6.13 | valid ppl 459.87
"""