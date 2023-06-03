# Import the datasets & necessary packages
from datasets import load_dataset

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
import re
import unicodedata
import itertools
import pandas as pd
import matplotlib.pyplot as plt

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
UNK_token = 3  # Unknown word token

# Preprocess the dataset
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
    "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
    "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
    "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
    "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
    "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
    "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
    "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
    "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
    "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
    "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
    "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
    "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
    "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
    "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
    "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
    "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
    "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
    "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
    "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
    "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
    "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
    "you're": "you are", "you've": "you have"}

# Define helper functions
def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub('"','', s)
    s = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in s.split(" ")])
    s = re.sub(r"'s\b","", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def LoadArticlesAndSummaries(dataset, type, count=0, max_article_length=0, max_summary_length=0):
    pairs = []
    
    if count > len(dataset[type]) or count == 0:
        count = len(dataset[type])
    
    # Choose articles and summaries with length less than max_article_length and max_summary_length
    i = 0
    num_sents = 0
    
    if (max_article_length == 0 or max_summary_length == 0):
        for i in range(count):
            article = normalizeString(dataset[type][i]['article'])
            summary = normalizeString(dataset[type][i]['highlights'])
            pairs.append([article, summary])
            
        return pairs
            
    for i in range(len(dataset[type])):
        if (num_sents >= count):
            break
        
        if (len(dataset[type][i]['article'].split()) <= max_article_length and len(dataset[type][i]['highlights'].split()) <= max_summary_length):
            pair = []
            pair.append(normalizeString(dataset['train'][i]['article']))
            pair.append(normalizeString(dataset['train'][i]['highlights']))
            pairs.append(pair)
            num_sents += 1
            # articles.append(normalizeString(dataset['train'][i]['article']))
            # summaries.append(normalizeString(dataset['train'][i]['highlights']))
        
    return pairs

# Create the vocabulary class
class Vocab(object):
    def __init__(self, pairs):
        super(Vocab, self).__init__()
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: "UNK"}
        self.num_words = 4  # Count SOS, EOS, PAD
        self.pairs = pairs
        
    def word2idx(self, word):
        if word in self.word2index:
            return self.word2index[word]
        else:
            return UNK_token
        
    def idx2word(self, idx):
        if idx in self.index2word:
            return self.index2word[idx]
        else:
            return self.index2word[UNK_token]
        
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)
            
    def addWord(self, word):
        if word in self.word2index:
            self.word2count[word] += 1
            
        else:
            self.word2index[word] = self.num_words
            self.index2word[self.num_words] = word
            self.word2count[word] = 1
            self.num_words += 1
            
    def buildVocab(self):        
        for pair in self.pairs:
            self.addSentence(pair[0])
            self.addSentence(pair[1])
            
        self.num_words = len(self.word2index)
        return self.num_words
    
    def trimVocab(self, min_count=0): 
        # Re-initialize dictionaries 
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: "UNK"}
        self.num_words = 4  # Count SOS, EOS, PAD
               
        for pair in self.pairs:
            self.addSentence(pair[0])
            self.addSentence(pair[1])

        # Remove words below a certain count threshold
        keep_words = []
        
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)
               
        # Re-initialize dictionaries 
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: "UNK"}
        self.num_words = 4  # Count SOS, EOS, PAD
        
        for word in keep_words:
            self.addWord(word)
            
        self.num_words = len(keep_words)
        return self.num_words

# Define utility functions
def trimRareWords(vocab, MIN_COUNT=0):
    init_num_words = vocab.buildVocab()
    final_num_words = vocab.trimVocab(MIN_COUNT)  
    print('Trimmed from {} words to {} words, removing {} words'.format(init_num_words, final_num_words, init_num_words - final_num_words))

def indexesFromSentence(vocab, sentence):
    return [vocab.word2idx(word) for word in sentence.split(' ')] + [EOS_token]

def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def tensorFromSentence(vocab, sentence):
    indexes = indexesFromSentence(vocab, sentence)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

def inputArticle(l, vocab):
    indexes_batch = [indexesFromSentence(vocab, article) for article in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths  

def outputSummary(l, vocab):
    indexes_batch = [indexesFromSentence(vocab, summary) for summary in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.BoolTensor(mask) 
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len 

# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputArticle(input_batch, voc)
    output, mask, max_target_len = outputSummary(output_batch, voc)
    return inp, lengths, output, mask, max_target_len

# Perform length analysis on the dataset
def data_analysis(pairs):
      article_word_count = []
      summary_word_count = []

      # populate the lists with sentence lengths
      for i in range(len(pairs)):
            article_word_count.append(len(pairs[i][0].split()))
            summary_word_count.append(len(pairs[i][1].split()))      

      length_df = pd.DataFrame({'text':article_word_count, 'summary': summary_word_count})
      length_df.hist(bins = 30)
      plt.show()

# Define the Encoder
class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        embedded = self.embedding(input_seq)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]

        return outputs, hidden

# Define the Encoder-Decoder Attention
class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        
    # using normal dot product attention
    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)
    
    def forward(self, hidden, encoder_outputs):
        attn_scores = self.dot_score(hidden, encoder_outputs)
        attn_scores = attn_scores.t()
        attn_dist = F.softmax(attn_scores, dim=1).unsqueeze(1)
        
        return attn_dist

# Define the decoder
class AttnDecoderRNN(nn.Module):
    def __init__(self, embedding, hidden_size, output_size, n_layers=1, dropout=0):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers==1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        
        self.attn = Attn(hidden_size)
        
    def forward(self, input_step, last_hidden, encoder_outputs):
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        
        rnn_output, hidden = self.gru(embedded, last_hidden)
        attn_scores = self.attn(rnn_output, encoder_outputs)
        
        context = attn_scores.bmm(encoder_outputs.transpose(0, 1))
        
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        
        return output, hidden

# Due to zero padding, masked NLL loss is used
def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()

# Define the training process
def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, batch_size, clip):

    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)
    
    # Lengths for rnn packing should always be on the cpu
    lengths = lengths.to("cpu")

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    for t in range(max_target_len):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
        # Teacher forcing: next input is current target
        decoder_input = target_variable[t].view(1, -1)
        # Calculate and accumulate loss
        mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
        loss += mask_loss
        print_losses.append(mask_loss.item() * nTotal)
        n_totals += nTotal

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals

# Epochs
def trainIters(voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, n_iteration, batch_size, print_every, clip):

    # Load batches for each iteration
    training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
                      for _ in range(n_iteration)]

    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0

    # Training loop
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # Run a training iteration with batch
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)
        print_loss += loss

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

# Load dataset
dataset = load_dataset("cnn_dailymail", '1.0.0')

# Build Article-Summary pairs
pairs = LoadArticlesAndSummaries(dataset, 'train', 40000, 256, 64)
len(pairs)

# Build Vocabulary with the trimmed dataset
vocab = Vocab(pairs)
trimRareWords(vocab, 10)

# Define the device
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
torch.cuda.get_device_name(torch.cuda.current_device())

# Empty the CUDA cache
torch.cuda.empty_cache()

# Configure the model
hidden_size = 300
encoder_n_layers = 1
decoder_n_layers = 1
dropout = 0.1
batch_size = 256

print('Building encoder and decoder ...')
embedding = nn.Embedding(vocab.num_words, hidden_size)
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = AttnDecoderRNN(embedding, hidden_size, vocab.num_words, decoder_n_layers, dropout)

encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')

# Train the model
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 100
print_every = 1

# Ensure dropout layers are in train mode
encoder.train()
decoder.train()

# Initialize optimizers
print('Building optimizers ...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

# If you have cuda, configure cuda to call
for state in encoder_optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()

for state in decoder_optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()

# Run training iterations
print("Starting Training!")
trainIters(vocab, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, n_iteration, batch_size, print_every, clip)
