export NEPOCHS=10
export MODEL=RNN_TANH
export NHID=200
export BATCH_SIZE=20
export BPTT=35
export NLAYERS=1

function train_hardcoded_examples() {
    NHID=400
    MODEL=GRU
    SAVE=model_epochs_10_model_${MODEL}_nhid_${NHID}_batch_size_20_bptt_35
    python main.py --device cuda:1 --epochs 10 --model $MODEL --nhid $NHID --batch_size 20 --bptt 35 \
        --save ${SAVE}.pt| tee ${SAVE}.md &

    NHID=200
    MODEL=LSTM
    SAVE=model_epochs_10_model_${MODEL}_nhid_${NHID}_batch_size_20_bptt_35
    python main.py --device cuda:1 --epochs 10 --model $MODEL --nhid $NHID --batch_size 20 --bptt 35 \
        --save ${SAVE}.pt| tee ${SAVE}.md &

    NHID=300
    MODEL=GRU
    SAVE=model_epochs_10_model_${MODEL}_nhid_${NHID}_batch_size_20_bptt_35
    python main.py --device cuda:1 --epochs 10 --model $MODEL --nhid $NHID --batch_size 20 --bptt 35 \
        --save ${SAVE}.pt| tee ${SAVE}.md
    # | end of epoch  10 | time: 40.47s | valid loss  5.33 | valid ppl   206.83
    # | End of training | test loss  5.10 | test ppl   164.42


    NHID=200
    MODEL=GRU
    SAVE=model_epochs_10_model_${MODEL}_nhid_${NHID}_batch_size_20_bptt_35
    python main.py --device cuda:1 --epochs 10 --model $MODEL --nhid $NHID --batch_size 20 --bptt 35 \
        --save ${SAVE}.pt| tee ${SAVE}.md
    # | end of epoch  10 | time: 37.21s | valid loss  6.13 | valid ppl   459.87
    # | End of training | test loss  5.27 | test ppl   195.38

    NHID=200
    MODEL=GRU
    SAVE=model_epochs_10_model_${MODEL}_nhid_${NHID}_batch_size_20_bptt_35
    python main.py --device cuda:1 --epochs 10 --model $MODEL --nhid $NHID --batch_size 20 --bptt 35 --save ${SAVE}.pt| tee ${SAVE}.md
    # model_epochs_10_model_LSTM_nhid_200_batch_size_20_bptt_35.md
    # | end of epoch  10 | time: 38.53s | valid loss  4.98 | valid ppl 145.29
    # | End of training | test loss  4.91 | test ppl   135.82


    python main.py --device cuda:0 --epochs 10 --model RNN_RELU --nhid 200 --batch_size 20 --bptt 35 --nlayers 2 --save model_epochs_10_model_RNN_RELU_nhid_200_batch_size_20_bptt_35_nlayers_2.pt
    # | end of epoch  10 | time: 35.40s | valid loss   nan | valid ppl      nan
    # | End of training | test loss   nan | test ppl      nan
}


function train_rnn_tanh() {
    NEPOCHS=10
    MODEL=RNN_TANH
    NHID=200
    BATCH_SIZE=20
    BPTT=35
    NLAYERS=1
    SAVE=model_epochs_${NEPOCHS}_model_${MODEL}_nhid_${NHID}_batch_size_${BATCH_SIZE}_bptt_${BPTT}_nlayers_${NLAYERS}
    echo "python main.py --cuda --epochs $NEPOCHS --model $MODEL --nhid $NHID --batch_size $BATCH_SIZE --bptt $BPTT --nlayers $NLAYERS --save ${SAVE}.pt" | tee -a ${SAVE}.md
    python main.py --cuda \
        --epochs $NEPOCHS \
        --model $MODEL \
        --nhid $NHID \
        --batch_size $BATCH_SIZE \
        --bptt $BPTT \
        --nlayers $NLAYERS \
        --save ${SAVE}.pt | tee ${SAVE}.md
    echo "python main.py --cuda --epochs $NEPOCHS --model $MODEL --nhid $NHID --batch_size $BATCH_SIZE --bptt $BPTT --nlayers $NLAYERS --save ${SAVE}.pt"  | tee -a ${SAVE}.md
    # | end of epoch  10 | time: 35.52s | valid loss  6.91 | valid ppl  1002.38
    # | End of training | test loss  6.85 | test ppl   941.55
    # python main.py --cuda --epochs 10 --model RNN_TANH --nhid 200 --batch_size 20 --bptt 35 --nlayers 1 --save model_epochs_10_model_RNN_TANH_nhid_200_batch_size_20_bptt_35_nlayers_1.pt >> model_epochs_10_model_RNN_TANH_nhid_200_batch_size_20_bptt_35_nlayers_1.md
}


export MODEL=RNN_TANH
export NLAYERS=1
export NEPOCHS=6
export NHID=200
export BATCH_SIZE=20
export BPTT=35

function train_model_layers_epochs() {
    MODEL=$1
    NLAYERS=$2
    NEPOCHS=$3
    SAVE=tuning_model_${MODEL}_nlayers_${NLAYERS}_epochs_${NEPOCHS}_nhid_${NHID}_batch_size_${BATCH_SIZE}_bptt_${BPTT}_nlayers
    echo "python main.py --cuda --model $MODEL --nlayers $NLAYERS --epochs $NEPOCHS --nhid $NHID --batch_size $BATCH_SIZE --bptt $BPTT --save ${SAVE}.pt" | tee -a ${SAVE}.md
    python main.py --cuda \
        --model $MODEL \
        --nlayers $NLAYERS \
        --epochs $NEPOCHS \
        --nhid $NHID \
        --batch_size $BATCH_SIZE \
        --bptt $BPTT \
        --save ${SAVE}.pt | tee ${SAVE}.md
    echo "python main.py --cuda --epochs $NEPOCHS --model $MODEL --nhid $NHID --batch_size $BATCH_SIZE --bptt $BPTT --nlayers $NLAYERS --save ${SAVE}.pt"  | tee -a ${SAVE}.md
    # | end of epoch  10 | time: 35.52s | valid loss  6.91 | valid ppl  1002.38
    # | End of training | test loss  6.85 | test ppl   941.55
    # python main.py --cuda --epochs 10 --model RNN_TANH --nhid 200 --batch_size 20 --bptt 35 --nlayers 1 --save model_epochs_10_model_RNN_TANH_nhid_200_batch_size_20_bptt_35_nlayers_1.pt >> model_epochs_10_model_RNN_TANH_nhid_200_batch_size_20_bptt_35_nlayers_1.md
}


function tune_model_layer_epochs() {

    train_model_layers_epochs "RNN_TANH" 1 6
    # | end of epoch   6 | time: 23.10s | valid loss  7.89 | valid ppl  2661.77 | End of training | test loss  7.70 | test ppl  2205.54

    train_model_layers_epochs "RNN_RELU" 1 6
    # | end of epoch   6 | time: 23.07s | valid loss   nan | valid ppl      nan | End of training | test loss   nan | test ppl      nan

    train_model_layers_epochs "LSTM" 1 6
    # | end of epoch   6 | time: 23.60s | valid loss  5.03 | valid ppl   152.69 | End of training | test loss  4.96 | test ppl   142.69  

    train_model_layers_epochs "GRU" 1 6
    # | end of epoch   6 | time: 23.44s | valid loss  4.94 | valid ppl   140.41 | End of training | test loss  4.87 | test ppl   130.37

    train_model_layers_epochs "RNN_TANH" 2 12
    train_model_layers_epochs "RNN_RELU" 2 12
    train_model_layers_epochs "LSTM" 2 12
    train_model_layers_epochs "GRU" 2 12
}
