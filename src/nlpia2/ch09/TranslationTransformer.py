# class Transformer
class TranslationTransformer(nn.Transformer):  # <2>
    def __init__(self,
            device=DEVICE,
            src_vocab_size: int = VOCAB_SIZE,
            src_pad_idx: int = PAD_IDX,
            tgt_vocab_size: int = VOCAB_SIZE,
            tgt_pad_idx: int = PAD_IDX,
            max_sequence_length: int = 100,
            d_model: int = 512,
            nhead: int = 8,
            num_encoder_layers: int = 6,
            num_decoder_layers: int = 6,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            activation: str = "relu"
        ):

        decoder_layer = CustomDecoderLayer(
            d_model, nhead, dim_feedforward,  # <3>
            dropout, activation)
        decoder_norm = nn.LayerNorm(d_model)
        decoder = CustomDecoder(
            decoder_layer, num_decoder_layers,
            decoder_norm)  # <4>

        super().__init__(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout, custom_decoder=decoder)

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.device = device

        self.src_emb = nn.Embedding(
            src_vocab_size, d_model)  # <5>
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)

        self.pos_enc = PositionalEncoding(
            d_model, dropout, max_sequence_length)  # <6>
        self.linear = nn.Linear(
            d_model, tgt_vocab_size)  # <7>
    def _make_key_padding_mask(self, t, pad_idx):
        mask = (t == pad_idx).to(self.device)
        return mask

    def prepare_src(self, src, src_pad_idx):
        src_key_padding_mask = self._make_key_padding_mask(
            src, src_pad_idx)
        src = rearrange(src, 'N S -> S N')
        src = self.pos_enc(self.src_emb(src)
            * math.sqrt(self.d_model))
        return src, src_key_padding_mask
    def prepare_tgt(self, tgt, tgt_pad_idx):
        tgt_key_padding_mask = self._make_key_padding_mask(
            tgt, tgt_pad_idx)
        tgt = rearrange(tgt, 'N T -> T N')
        tgt_mask = self.generate_square_subsequent_mask(
            tgt.shape[0]).to(self.device)
        tgt = self.pos_enc(self.tgt_emb(tgt)
            * math.sqrt(self.d_model))
        return tgt, tgt_key_padding_mask, tgt_mask
    def forward(self, src, tgt):
        src, src_key_padding_mask = self.prepare_src(
            src, self.src_pad_idx)
        tgt, tgt_key_padding_mask, tgt_mask = self.prepare_tgt(
            tgt, self.tgt_pad_idx)
        memory_key_padding_mask = src_key_padding_mask.clone()
        output = super().forward(
            src, tgt, tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask)
        output = rearrange(output, 'T N E -> N T E')
        return self.linear(output)
    def init_weights(self):
        def _init_weights(m):
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight.data)
        self.apply(_init_weights);  # <1>
