from EncoderDecoder import subsequent_mask
from FullModel import make_model
import torch.nn.functional as F
import torch

def inference_test():
    test_model = make_model(11, 11, 2)
    test_model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)

    memory = test_model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src)

    for i in range(9):
        out = test_model.decode(memory, src_mask, ys,
        subsequent_mask(ys.size(1)).type_as(src.data))
        prob = test_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src).fill_(next_word)], dim=1
        )
    
    print("Example Untrained Model Output: ", ys)

def run_tests():
    for test in range(10):
        inference_test()

run_tests()

