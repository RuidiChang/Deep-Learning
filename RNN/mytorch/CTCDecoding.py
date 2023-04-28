import numpy as np

class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        
        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set


    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path = []
        blank = 0
        path_prob = 1

        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        for s in range(len(y_probs[0])):
        # 2. Iterate over symbol probabilities
            for p in range(len(y_probs[0][s])):
        # 3. update path probability, by multiplying with the current max probability
                path_prob *= np.max(y_probs[:,s,p])
        # 4. Select most probable symbol and append to decoded_path
                index = np.argmax(y_probs[:,s,p])
        # 5. Compress sequence (Inside or outside the loop)
                if index==0:
                    blank = 1
                else:
                    if blank:
                        decoded_path.append(self.symbol_set[index - 1])
                        blank = 0
                    else:
                        if len(decoded_path) == 0 or decoded_path[-1] != self.symbol_set[index - 1]:
                            decoded_path.append(self.symbol_set[index - 1])
        decoded_path = ''.join(decoded_path)
        return decoded_path, path_prob
        


class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width

    def decode(self, y_probs):
        """
        
        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
			batch size for part 1 will remain 1, but if you plan to use your
			implementation for part 2 you need to incorporate batch_size

        Returns
        -------
        
        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """

        T = y_probs.shape[1]
        bestPath, FinalPathScore = None, None
        
        
        symbol_score = {}
        blank_score = {}
        logits = y_probs[:, :, 0]

        blank_score[None] = logits[0, 0]
        for i in range(len(self.symbol_set)):
            symbol_score[self.symbol_set[i]] = logits[i+1, 0]

        for t in range(1, logits.shape[1]):
            scorelist = list(symbol_score.values())+list(blank_score.values())
            scorelist.sort(reverse=True)
            if self.beam_width - 1 < len(scorelist):
                cutoff = scorelist[self.beam_width-1]
            else:
                cutoff = scorelist[-1]
            symbol_path_score = {key: val for key, val in symbol_score.items() if val >= cutoff}
            symbolPath = set(symbol_path_score.keys())
            blank_path_score = {key: val for key, val in blank_score.items() if val >= cutoff}
            blankPath = set(blank_path_score.keys())

            new_symbol_path = set()
            symbol_score = {}
            for path in blankPath:
                for i in range(len(self.symbol_set)):
                    new_path = path + self.symbol_set[i] if path != None else self.symbol_set[i]
                    new_symbol_path.add(new_path)
                    symbol_score[new_path] = blank_path_score[path]*logits[i+1, t]
            for path in symbolPath:
                for i in range(len(self.symbol_set)):
                    if self.symbol_set[i] == path[-1]:
                        new_path = path
                    else:
                      new_path = path + self.symbol_set[i]
                    if new_path in new_symbol_path:
                        symbol_score[new_path] += symbol_path_score[path]*logits[i+1, t]
                    else:
                        new_symbol_path.add(new_path)
                        symbol_score[new_path] = symbol_path_score[path]*logits[i+1, t]

            new_blank_path = set()
            blank_score = {}
            for path in blankPath:
                new_blank_path.add(path)
                blank_score[path] = blank_path_score[path]*logits[0, t]
            for path in symbolPath:
                if path in new_blank_path:
                    blank_score[path] += symbol_path_score[path]*logits[0, t]
                else:
                    new_blank_path.add(path)
                    blank_score[path] = symbol_path_score[path]*logits[0, t]

        merged_paths = new_symbol_path
        FinalPathScore = symbol_score
        for path in new_blank_path:
            if path in merged_paths:
                FinalPathScore[path] += blank_score[path]
            else:
                merged_paths.add(path)
                FinalPathScore[path] = blank_score[path]
        
        bestPath = max(FinalPathScore, key=FinalPathScore.get)

        return bestPath, FinalPathScore
