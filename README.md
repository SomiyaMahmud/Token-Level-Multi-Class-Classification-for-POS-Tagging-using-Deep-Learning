# Token-Level Multi-Class Classification for POS Tagging using Deep Learning

This project focuses on token-level multi-class classification for Part-of-Speech (POS) tagging, a foundational task in Natural Language Processing (NLP) where each word in a sentence is labeled with its appropriate part of speech. The objective is to assess how different deep learning models, particularly Recurrent Neural Network (RNN) variants, perform on this task. Four architectures were implemented and evaluated: RNN, Long Short-Term Memory (LSTM), Gated Recurrent Unit (GRU), and Bidirectional LSTM (BiLSTM). The goal is to analyze which architecture provides the best trade-off between accuracy and model complexity. The models were trained and tested on a pre-split POS tagging dataset and evaluated using accuracy, F1-score, confusion matrix, and classification report.

## Dataset

<p>
The dataset used for training and evaluating the model is a curated POS tagging dataset, containing tokenized sentences and their corresponding Part-of-Speech (POS) tags.
</p>

<h3>Dataset Format</h3>

<p>
The dataset is stored in CSV format and split into two files:
<ul>
  <li><code>train.csv</code>: Used for model training and validation.</li>
  <li><code>test.csv</code>: Used for final model evaluation.</li>
</ul>
</p>

<p>
Each file contains two key columns:
<ul>
  <li><strong>Sentence</strong>: A complete sentence represented as a space-separated string of words.</li>
  <li><strong>POS</strong>: A list of the corresponding Part-of-Speech tags for each word in the sentence.</li>
</ul>
</p>


<h3>Preprocessing</h3>

<ul>
  <li><strong>Tokenization</strong>: Split each sentence into individual word tokens.</li>
  <li><strong>Vocabulary Building</strong>: Assigned unique indices to each word using a tokenizer.</li>
  <li><strong>POS Tag Encoding</strong>: Converted each POS tag into a corresponding numeric label using label-based indexing.</li>
  <li><strong>Padding</strong>:
    <ul>
      <li>Applied padding to both token sequences and POS tag sequences to ensure uniform input lengths.</li>
      <li>Padding was done using 0s to match the maximum sequence length.</li>
    </ul>
  </li>
  <li><strong>Train-Validation Split</strong>: 
    <ul>
      <li>After preprocessing, the training data was split into 90% for training and 10% for validation.</li>
      <li>This helped monitor model performance and reduce overfitting.</li>
    </ul>
  </li>
</ul>



## Model Architecture


<p>The models consist of multiple layers designed for sequence labeling, with each model differing only in the type of recurrent layer used:</p>

<ul>
  <li><strong>Embedding Layer</strong>: Converts input tokens into 128-dimensional trainable vector representations, allowing the model to capture word semantics.</li>

  <li><strong>Recurrent Layer</strong>: Depending on the model variant, one of the following is used:
    <ul>
      <li><strong>Simple RNN</strong>: Basic recurrent layer that processes tokens sequentially.</li>
      <li><strong>LSTM</strong>: Long Short-Term Memory layer capable of learning long-range dependencies through gating mechanisms.</li>
      <li><strong>GRU</strong>: Gated Recurrent Unit layer, offering similar capabilities to LSTM with fewer parameters.</li>
      <li><strong>Bidirectional LSTM</strong>: Enhances context understanding by processing sequences in both forward and backward directions.</li>
    </ul>
    All recurrent layers are configured with:
    <ul>
      <li>64 hidden units</li>
      <li>return_sequences=True to predict a tag for each token</li>
      <li>dropout=0.3 and recurrent_dropout=0.3 for regularization</li>
    </ul>
  </li>

  <li><strong>Dropout Layer</strong>: An additional dropout layer (0.3) is applied after the recurrent layer to reduce overfitting.</li>

  <li><strong>TimeDistributed Dense Layer</strong>: A fully connected layer wrapped with <code>TimeDistributed</code>, applying a softmax activation to output a probability distribution over all POS tags for each token in the sequence.</li>

  <li><strong>Compilation & Training</strong>:
    <ul>
      <li><strong>Loss Function</strong>: categorical_crossentropy</li>
      <li><strong>Optimizer</strong>: Adam with a learning rate of 0.001</li>
      <li><strong>Metrics</strong>: Accuracy</li>
      <li><strong>Training Configuration</strong>:
        <ul>
          <li>Epochs: 20</li>
          <li>Batch size: 32</li>
        </ul>
      </li>
    </ul>
  </li>
</ul>

<p>Each of these model variants was trained on the same data and evaluated using consistent metrics for fair comparison.</p>


## Results:

<p>To evaluate the models, we computed several standard metrics:</p>
<ul>
  <li><strong>Accuracy:</strong> Token-level accuracy was calculated on the test set.</li>
  <li><strong>F1-Score:</strong> Both macro and weighted F1-scores were used to evaluate balance across all POS tags.</li>
  <li><strong>Classification Report:</strong> Included precision, recall, and F1-score for each individual tag. </li>
  <li><strong>Confusion Matrix</li>
</ul>

<table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; text-align: center;">
  <thead>
    <tr>
      <th>Model</th>
      <th>Accuracy</th>
      <th>Macro F1 Score</th>
      <th>Weighted F1 Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Simple RNN</td>
      <td>0.9575</td>
      <td>0.8976</td>
      <td>0.9576</td>
    </tr>
    <tr>
      <td>LSTM</td>
      <td>0.9598</td>
      <td>0.9105</td>
      <td>0.9599</td>
    </tr>
    <tr>
      <td>GRU</td>
      <td>0.9600</td>
      <td>0.9143</td>
      <td>0.9602</td>
    </tr>
    <tr>
      <td>BiLSTM</td>
      <td>0.9687</td>
      <td>0.9479</td>
      <td>0.9688</td>
    </tr>
  </tbody>
</table>

<h2>Improvements</h2>

<p>Future work may involve:</p>

<ul>
  <li>Explore pretrained embeddings such as GloVe, and BERT to enhance semantic understanding and model performance.</li>
  <li>Investigate transformer-based architectures like BERT to capture long-range dependencies and contextual information more effectively.</li>
  <li>Integrate character-level embeddings to improve recognition of rare or unknown tokens.</li>
  <li>Apply the methodology to multilingual and domain-specific datasets to boost generalizability and adaptability.</li>
</ul>
