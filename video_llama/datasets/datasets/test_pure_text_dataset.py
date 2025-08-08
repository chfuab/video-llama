sys_prompt = '''You are given a passage, a question, and three options of answers to the question indexed by A, B, C. Your task is to select the correct answer to the question from the three options according to the passage.'''
passages = ['There are originally 4 balls in a box. 2 balls are then taken away from the box.', 
            'There are originally 5 balls in a box. 4 balls are then taken away from the box.', 
            'There are originally 9 balls in a box. 3 balls are then taken away from the box.', 
            'There are originally 7 balls in a box. 2 balls are then taken away from the box.']
question = "How many balls are finally left in the box?"
options = ['A. 3\nB. 2\n C. 7', 'A. 1\nB. 5\n C. 3', 'A. 3\nB. 5\n C. 6', 'A. 5\nB. 1\n C. 4']
correct_answer = ['B', 'A', 'C', 'A']

class Textdataset:
    def __init__(self):
        self.sys_prompt = sys_prompt
        self.passages = passages
        self.question = question
        self.options = options
        self.correct_answer = correct_answer
        
    def __getitem__(self, idx):
        prompt_all = self.sys_prompt + '\n' + 'passage:' + self.passages[idx] + '\n' + 'question: ' + self.question + '\n' + 'options:' + self.options[idx] + '\n' + 'correct answer:'
        ground_truth = self.correct_answer[idx]
        return {"prompt_all": prompt_all, "correct_ans_id": ground_truth}
    def __len__(self):
        return len(self.passages)