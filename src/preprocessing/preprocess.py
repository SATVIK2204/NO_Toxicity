from nltk import wordpunct_tokenize
from torch.utils.data import Dataset
from collections import Counter

def tokenize(text):
    """Turn text into discrete tokens.

    Remove tokens that are not words.
    """
    text = text.lower()
    tokens = wordpunct_tokenize(text)

    # Only keep words
    tokens = [token for token in tokens
              if all(char.isalpha() for char in token)]

    return tokens


class PrepareTheData(Dataset):
    def __init__(self, df, labels, max_vocab=1000):
        self.max_vocab = max_vocab
        
        # Extra tokens to add
        self.padding_token = '<PAD>'
        self.unknown_word_token='<UNK>'
        # Helper function
        self.flatten = lambda x: [sublst for lst in x for sublst in lst]
        self.df_old=df.copy()

        df=self.max_words(df)
        self.df_new=df.copy()
        # Tokenize inputs (English) and targets (French)
        self.tokenize_df(df)
        print('Tokenization Done')

        # To reduce computational complexity, replace rare words with <UNK>
        self.replace_rare_tokens(df)
        print('Replacing Done')

        # # Prepare variables with mappings of tokens to indices
        self.create_token2idx(df)
        print('token2idx created')
        

        # Remove sequences with mostly <UNK>
        df = self.remove_mostly_unk(df)       
       

        # Convert tokens to indices
        self.tokens_to_indices(df)
        print('tokens_to_indices created')
        print('Presprocessing done')

        self.labels=labels
    
    def tokenize_df(self, df):
        """Turn inputs and targets into tokens."""
        df.loc[:,'comment_text'] = df.comment_text.apply(tokenize)

    def max_words(self, df, max_length=100):
        df = df[df['comment_text'].str.split().str.len().lt(max_length)]
        return df
        
        
    def get_most_common_tokens(self, tokens_series):
        """Return the max_vocab most common tokens."""
        all_tokens = self.flatten(tokens_series)
        # Substract 4 for <PAD>, <SOS>, <EOS>, and <UNK>
        common_tokens = set(list(zip(*Counter(all_tokens).most_common(
            self.max_vocab - 2)))[0])

        return common_tokens

    def replace_rare_tokens(self, df):
        """Replace rare tokens with <UNK>."""
        common_tokens_inputs = self.get_most_common_tokens(
            df.comment_text.tolist(),
        )
        
        df.loc[:, 'comment_text'] = df.comment_text.apply(
            lambda tokens: [token if token in common_tokens_inputs 
                            else self.unknown_word_token for token in tokens]
        )
        
    def remove_mostly_unk(self, df, threshold=0.8):
        """Remove sequences with mostly <UNK>."""
        calculate_ratio = (
            lambda tokens: sum(1 for token in tokens if token != '<UNK>')
            / len(tokens) > threshold
        )
        df = df[df.comment_text.apply(calculate_ratio)]
        return df
        
    def create_token2idx(self, df):
        """Create variables with mappings from tokens to indices."""
        unique_tokens_inputs = set(self.flatten(df.comment_text))
        
        
        for token in reversed([
            self.padding_token,
            self.unknown_word_token,
        ]):
            if token in unique_tokens_inputs:
                unique_tokens_inputs.remove(token)
            
                
        unique_tokens_inputs = sorted(list(unique_tokens_inputs))
        

        # Add <PAD>, <SOS>, <EOS>, and <UNK> tokens
        for token in reversed([
            self.padding_token,
            self.unknown_word_token,
        ]):
            
            unique_tokens_inputs = [token] + unique_tokens_inputs
            
            
        self.token2idx_inputs = {token: idx for idx, token
                                 in enumerate(unique_tokens_inputs)}
        self.idx2token_inputs = {idx: token for token, idx
                                 in self.token2idx_inputs.items()}
            
        
    def tokens_to_indices(self, df):
        """Convert tokens to indices."""
        df.loc[:,'comment_text'] = df.comment_text.apply(
            lambda tokens: [self.token2idx_inputs[token] for token in tokens])

             
        self.indices_pairs = list(df.comment_text)
    
    def __getitem__(self, idx):
        """Return example at index idx."""
        return self.indices_pairs[idx], self.labels.to_numpy()[idx]
        
    def __len__(self):
        return len(self.indices_pairs)