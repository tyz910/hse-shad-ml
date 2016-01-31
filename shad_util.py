import os


class AnswerPrinter:
    def __init__(self):
        self.files = {}

    def print_answer(self, num, line, nl=False):
        print line

        if num not in self.files:
            f = open(os.getcwd() + '/answers/a' + str(num) + '.txt', 'w+')
            self.files[num] = f
        else:
            f = self.files[num]

        if nl:
            line += '\n'

        f.write(line)


printer = AnswerPrinter()
print_answer = printer.print_answer
