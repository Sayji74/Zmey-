# -*- coding: utf-8 -*-
#######################################
# ИМПОРТЫ
#######################################

from строки_со_стрелками import string_with_arrows
import string
import os
import math

#######################################
# КОНСТАНТЫ
#######################################

DIGITS = '0123456789'
LETTERS = string.ascii_letters + 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
LETTERS_DIGITS = LETTERS + DIGITS

#######################################
# ОШИБКИ
#######################################

class Error:
    def __init__(self, pos_start, pos_end, error_name, details):
        self.pos_start = pos_start
        self.pos_end = pos_end
        self.error_name = error_name
        self.details = details
    
    def as_string(self):
        result = f'{self.error_name}: {self.details}\n'
        result += f'Файл {self.pos_start.fn}, строка {self.pos_start.ln + 1}'
        result += '\n\n' + string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)
        return result

class IllegalCharError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, 'Недопустимый символ', details)

class ExpectedCharError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, 'Ожидаемый символ', details)

class InvalidSyntaxError(Error):
    def __init__(self, pos_start, pos_end, details=''):
        super().__init__(pos_start, pos_end, 'Неверный синтаксис', details)

class RTError(Error):
    def __init__(self, pos_start, pos_end, details, context):
        super().__init__(pos_start, pos_end, 'Ошибка выполнения', details)
        self.context = context

    def as_string(self):
        result = self.generate_traceback()
        result += f'{self.error_name}: {self.details}'
        result += '\n\n' + string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)
        return result

    def generate_traceback(self):
        result = ''
        pos = self.pos_start
        ctx = self.context

        while ctx:
            result = f'  Файл {pos.fn}, строка {str(pos.ln + 1)}, в {ctx.display_name}\n' + result
            pos = ctx.parent_entry_pos
            ctx = ctx.parent

        return 'Трассировка (последний вызов последним):\n' + result

#######################################
# ПОЗИЦИЯ
#######################################

class Position:
    def __init__(self, idx, ln, col, fn, ftxt):
        self.idx = idx
        self.ln = ln
        self.col = col
        self.fn = fn
        self.ftxt = ftxt

    def advance(self, current_char=None):
        self.idx += 1
        self.col += 1

        if current_char == '\n':
            self.ln += 1
            self.col = 0

        return self

    def copy(self):
        return Position(self.idx, self.ln, self.col, self.fn, self.ftxt)

#######################################
# ТОКЕНЫ
#######################################

TT_INT        = 'ЦЕЛОЕ'
TT_FLOAT      = 'ДРОБНОЕ'
TT_STRING     = 'СТРОКА'
TT_IDENTIFIER = 'ИДЕНТИФИКАТОР'
TT_KEYWORD    = 'КЛЮЧЕВОЕ_СЛОВО'
TT_PLUS       = 'ПЛЮС'
TT_MINUS      = 'МИНУС'
TT_MUL        = 'УМНОЖИТЬ'
TT_DIV        = 'ДЕЛИТЬ'
TT_POW        = 'СТЕПЕНЬ'
TT_EQ         = 'РАВНО'
TT_LPAREN     = 'ЛЕВАЯ_СКОБКА'
TT_RPAREN     = 'ПРАВАЯ_СКОБКА'
TT_LSQUARE    = 'ЛЕВАЯ_КВАДРАТНАЯ'
TT_RSQUARE    = 'ПРАВАЯ_КВАДРАТНАЯ'
TT_EE         = 'РАВНО_РАВНО'
TT_NE         = 'НЕ_РАВНО'
TT_LT         = 'МЕНЬШЕ'
TT_GT         = 'БОЛЬШЕ'
TT_LTE        = 'МЕНЬШЕ_ИЛИ_РАВНО'
TT_GTE        = 'БОЛЬШЕ_ИЛИ_РАВНО'
TT_COMMA      = 'ЗАПЯТАЯ'
TT_ARROW      = 'СТРЕЛКА'
TT_NEWLINE    = 'НОВАЯ_СТРОКА'
TT_EOF        = 'КОНЕЦ_ФАЙЛА'

KEYWORDS = [
    'ВЪ',          # VAR
    'ДА',          # AND
    'ЛИБО',        # OR
    'НЕ',          # NOT
    'ЕСЛИ',        # IF
    'КОЛИ',        # ELIF
    'ИЛИ',         # ELSE
    'ПО',          # FOR
    'ДО',          # TO
    'ШАГ',         # STEP
    'ПОКА',        # WHILE
    'ВЕСЕЛЬЕ',     # FUN
    'ПОСЛЕ',       # THEN
    'ЗАКОНЧИТЬ',   # END
    'ВЕРНИСЬ-КА',  # RETURN
    'ПРОДОЛЖАЙ',   # CONTINUE
    'ТОРМОЗИ',     # BREAK
]

class Token:
    def __init__(self, type_, value=None, pos_start=None, pos_end=None):
        self.type = type_
        self.value = value

        if pos_start:
            self.pos_start = pos_start.copy()
            self.pos_end = pos_start.copy()
            self.pos_end.advance()

        if pos_end:
            self.pos_end = pos_end.copy()

    def matches(self, type_, value):
        return self.type == type_ and self.value == value
    
    def __repr__(self):
        if self.value: return f'{self.type}:{self.value}'
        return f'{self.type}'

#######################################
# ЛЕКСЕР
#######################################

class Lexer:
    def __init__(self, fn, text):
        self.fn = fn
        self.text = text
        self.pos = Position(-1, 0, -1, fn, text)
        self.current_char = None
        self.advance()
    
    def advance(self):
        self.pos.advance(self.current_char)
        self.current_char = self.text[self.pos.idx] if self.pos.idx < len(self.text) else None

    def make_tokens(self):
        tokens = []

        while self.current_char is not None:
            if self.current_char in ' \t':
                self.advance()
            elif self.current_char == '#':
                self.skip_comment()
            elif self.current_char in ';\n':
                tokens.append(Token(TT_NEWLINE, pos_start=self.pos))
                self.advance()
            elif self.current_char in DIGITS:
                tokens.append(self.make_number())
            elif self.current_char in LETTERS:
                tokens.append(self.make_identifier())
            elif self.current_char == '"':
                tokens.append(self.make_string())
            elif self.current_char == '+':
                tokens.append(Token(TT_PLUS, pos_start=self.pos))
                self.advance()
            elif self.current_char == '-':
                tokens.append(self.make_minus_or_arrow())
            elif self.current_char == '*':
                tokens.append(Token(TT_MUL, pos_start=self.pos))
                self.advance()
            elif self.current_char == '/':
                tokens.append(Token(TT_DIV, pos_start=self.pos))
                self.advance()
            elif self.current_char == '^':
                tokens.append(Token(TT_POW, pos_start=self.pos))
                self.advance()
            elif self.current_char == '(':
                tokens.append(Token(TT_LPAREN, pos_start=self.pos))
                self.advance()
            elif self.current_char == ')':
                tokens.append(Token(TT_RPAREN, pos_start=self.pos))
                self.advance()
            elif self.current_char == '[':
                tokens.append(Token(TT_LSQUARE, pos_start=self.pos))
                self.advance()
            elif self.current_char == ']':
                tokens.append(Token(TT_RSQUARE, pos_start=self.pos))
                self.advance()
            elif self.current_char == '!':
                token, error = self.make_not_equals()
                if error: return [], error
                tokens.append(token)
            elif self.current_char == '=':
                tokens.append(self.make_equals())
            elif self.current_char == '<':
                tokens.append(self.make_less_than())
            elif self.current_char == '>':
                tokens.append(self.make_greater_than())
            elif self.current_char == ',':
                tokens.append(Token(TT_COMMA, pos_start=self.pos))
                self.advance()
            else:
                pos_start = self.pos.copy()
                char = self.current_char
                self.advance()
                return [], IllegalCharError(pos_start, self.pos, "'" + char + "'")

        tokens.append(Token(TT_EOF, pos_start=self.pos))
        return tokens, None

    def make_number(self):
        num_str = ''
        dot_count = 0
        pos_start = self.pos.copy()

        while self.current_char is not None and self.current_char in DIGITS + '.':
            if self.current_char == '.':
                if dot_count == 1: break
                dot_count += 1
            num_str += self.current_char
            self.advance()

        if dot_count == 0:
            return Token(TT_INT, int(num_str), pos_start, self.pos)
        else:
            return Token(TT_FLOAT, float(num_str), pos_start, self.pos)

    def make_string(self):
        string = ''
        pos_start = self.pos.copy()
        escape_character = False
        self.advance()

        escape_characters = {
            'n': '\n',
            't': '\t'
        }

        while self.current_char is not None and (self.current_char != '"' or escape_character):
            if escape_character:
                string += escape_characters.get(self.current_char, self.current_char)
                escape_character = False
            else:
                if self.current_char == '\\':
                    escape_character = True
                else:
                    string += self.current_char
            self.advance()

        if self.current_char == '"':
            self.advance()  # Пропускаем закрывающую кавычку
        return Token(TT_STRING, string, pos_start, self.pos)

    def make_identifier(self):
        id_str = ''
        pos_start = self.pos.copy()

        while self.current_char is not None and self.current_char in LETTERS_DIGITS + '_':
            id_str += self.current_char
            self.advance()

        id_str = id_str.upper()  # Приводим к верхнему регистру
        tok_type = TT_KEYWORD if id_str in KEYWORDS else TT_IDENTIFIER
        return Token(tok_type, id_str, pos_start, self.pos)

    def skip_comment(self):
        self.advance()
        while self.current_char is not None and self.current_char != '\n':
            self.advance()

    def make_minus_or_arrow(self):
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == '>':
            self.advance()
            return Token(TT_ARROW, pos_start=pos_start, pos_end=self.pos)
        return Token(TT_MINUS, pos_start=pos_start)

    def make_not_equals(self):
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == '=':
            self.advance()
            return Token(TT_NE, pos_start=pos_start, pos_end=self.pos), None
        return None, ExpectedCharError(pos_start, self.pos, "'=' (после '!')")

    def make_equals(self):
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == '=':
            self.advance()
            return Token(TT_EE, pos_start=pos_start, pos_end=self.pos)
        return Token(TT_EQ, pos_start=pos_start)

    def make_less_than(self):
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == '=':
            self.advance()
            return Token(TT_LTE, pos_start=pos_start, pos_end=self.pos)
        return Token(TT_LT, pos_start=pos_start)

    def make_greater_than(self):
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == '=':
            self.advance()
            return Token(TT_GTE, pos_start=pos_start, pos_end=self.pos)
        return Token(TT_GT, pos_start=pos_start)

#######################################
# УЗЛЫ AST
#######################################

class NumberNode:
    def __init__(self, tok):
        self.tok = tok
        self.pos_start = tok.pos_start
        self.pos_end = tok.pos_end

    def __repr__(self):
        return f'{self.tok}'

class StringNode:
    def __init__(self, tok):
        self.tok = tok
        self.pos_start = tok.pos_start
        self.pos_end = tok.pos_end

    def __repr__(self):
        return f'{self.tok}'

class ListNode:
    def __init__(self, element_nodes, pos_start, pos_end):
        self.element_nodes = element_nodes
        self.pos_start = pos_start
        self.pos_end = pos_end

class VarAccessNode:
    def __init__(self, var_name_tok):
        self.var_name_tok = var_name_tok
        self.pos_start = var_name_tok.pos_start
        self.pos_end = var_name_tok.pos_end

class VarAssignNode:
    def __init__(self, var_name_tok, value_node):
        self.var_name_tok = var_name_tok
        self.value_node = value_node
        self.pos_start = var_name_tok.pos_start
        self.pos_end = value_node.pos_end

class BinOpNode:
    def __init__(self, left_node, op_tok, right_node):
        self.left_node = left_node
        self.op_tok = op_tok
        self.right_node = right_node
        self.pos_start = left_node.pos_start
        self.pos_end = right_node.pos_end

    def __repr__(self):
        return f'({self.left_node}, {self.op_tok}, {self.right_node})'

class UnaryOpNode:
    def __init__(self, op_tok, node):
        self.op_tok = op_tok
        self.node = node
        self.pos_start = op_tok.pos_start
        self.pos_end = node.pos_end

    def __repr__(self):
        return f'({self.op_tok}, {self.node})'

class IfNode:
    def __init__(self, cases, else_case):
        self.cases = cases
        self.else_case = else_case
        self.pos_start = cases[0][0].pos_start
        self.pos_end = (else_case or cases[-1])[0].pos_end

class ForNode:
    def __init__(self, var_name_tok, start_value_node, end_value_node, step_value_node, body_node, should_return_null):
        self.var_name_tok = var_name_tok
        self.start_value_node = start_value_node
        self.end_value_node = end_value_node
        self.step_value_node = step_value_node
        self.body_node = body_node
        self.should_return_null = should_return_null
        self.pos_start = var_name_tok.pos_start
        self.pos_end = body_node.pos_end

class WhileNode:
    def __init__(self, condition_node, body_node, should_return_null):
        self.condition_node = condition_node
        self.body_node = body_node
        self.should_return_null = should_return_null
        self.pos_start = condition_node.pos_start
        self.pos_end = body_node.pos_end

class FuncDefNode:
    def __init__(self, var_name_tok, arg_name_toks, body_node, should_auto_return):
        self.var_name_tok = var_name_tok
        self.arg_name_toks = arg_name_toks
        self.body_node = body_node
        self.should_auto_return = should_auto_return
        self.pos_start = var_name_tok.pos_start if var_name_tok else body_node.pos_start
        self.pos_end = body_node.pos_end

class CallNode:
    def __init__(self, node_to_call, arg_nodes):
        self.node_to_call = node_to_call
        self.arg_nodes = arg_nodes
        self.pos_start = node_to_call.pos_start
        self.pos_end = node_to_call.pos_end if not arg_nodes else arg_nodes[-1].pos_end

class ReturnNode:
    def __init__(self, node_to_return, pos_start, pos_end):
        self.node_to_return = node_to_return
        self.pos_start = pos_start
        self.pos_end = pos_end

class ContinueNode:
    def __init__(self, pos_start, pos_end):
        self.pos_start = pos_start
        self.pos_end = pos_end

class BreakNode:
    def __init__(self, pos_start, pos_end):
        self.pos_start = pos_start
        self.pos_end = pos_end

#######################################
# РЕЗУЛЬТАТ ПАРСЕРА
#######################################

class ParseResult:
    def __init__(self):
        self.error = None
        self.node = None
        self.advance_count = 0
        self.to_reverse_count = 0

    def register_advancement(self):
        self.advance_count += 1

    def register(self, res):
        self.advance_count += res.advance_count
        if res.error: self.error = res.error
        return res.node

    def try_register(self, res):
        if res.error:
            self.to_reverse_count = res.advance_count
            return None
        return self.register(res)

    def success(self, node):
        self.node = node
        return self

    def failure(self, error):
        if not self.error or self.advance_count == 0:
            self.error = error
        return self

#######################################
# ПАРСЕР
#######################################

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.tok_idx = -1
        self.current_tok = None
        self.advance()

    def advance(self):
        self.tok_idx += 1
        self.update_current_tok()
        return self.current_tok

    def reverse(self, amount=1):
        self.tok_idx -= amount
        self.update_current_tok()
        return self.current_tok

    def update_current_tok(self):
        if self.tok_idx >= 0 and self.tok_idx < len(self.tokens):
            self.current_tok = self.tokens[self.tok_idx]

    def parse(self):
        res = self.statements()
        if not res.error and self.current_tok.type != TT_EOF:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Ожидался '+', '-', '*', '/', '^', '==', '!=', '<', '>', '<=', '>=', 'ДА' или 'ЛИБО'"
            ))
        return res

    def statements(self):
        res = ParseResult()
        statements = []
        pos_start = self.current_tok.pos_start.copy()

        while self.current_tok.type == TT_NEWLINE:
            res.register_advancement()
            self.advance()

        statement = res.register(self.statement())
        if res.error: return res
        statements.append(statement)

        more_statements = True

        while True:
            newline_count = 0
            while self.current_tok.type == TT_NEWLINE:
                res.register_advancement()
                self.advance()
                newline_count += 1
            if newline_count == 0:
                more_statements = False

            if not more_statements: break
            statement = res.try_register(self.statement())
            if not statement:
                self.reverse(res.to_reverse_count)
                more_statements = False
                continue
            statements.append(statement)

        return res.success(ListNode(statements, pos_start, self.current_tok.pos_end.copy()))

    def statement(self):
        res = ParseResult()
        pos_start = self.current_tok.pos_start.copy()

        if self.current_tok.matches(TT_KEYWORD, 'ВЕРНИСЬ-КА'):
            res.register_advancement()
            self.advance()

            expr = res.try_register(self.expr())
            if not expr:
                self.reverse(res.to_reverse_count)
            return res.success(ReturnNode(expr, pos_start, self.current_tok.pos_end.copy()))

        if self.current_tok.matches(TT_KEYWORD, 'ПРОДОЛЖАЙ'):
            res.register_advancement()
            self.advance()
            return res.success(ContinueNode(pos_start, self.current_tok.pos_end.copy()))

        if self.current_tok.matches(TT_KEYWORD, 'ТОРМОЗИ'):
            res.register_advancement()
            self.advance()
            return res.success(BreakNode(pos_start, self.current_tok.pos_end.copy()))

        expr = res.register(self.expr())
        if res.error:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Ожидалось 'ВЕРНИСЬ-КА', 'ПРОДОЛЖАЙ', 'ТОРМОЗИ', 'ВЪ', 'ЕСЛИ', 'ПО', 'ПОКА', 'ВЕСЕЛЬЕ', целое, дробное, идентификатор, '[', '(', '+' или '-'"
            ))
        return res.success(expr)

    def expr(self):
        res = ParseResult()

        if self.current_tok.matches(TT_KEYWORD, 'ВЪ'):
            res.register_advancement()
            self.advance()

            if self.current_tok.type != TT_IDENTIFIER:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Ожидался идентификатор"
                ))

            var_name = self.current_tok
            res.register_advancement()
            self.advance()

            if self.current_tok.type != TT_EQ:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Ожидался '='"
                ))

            res.register_advancement()
            self.advance()
            expr = res.register(self.expr())
            if res.error: return res
            return res.success(VarAssignNode(var_name, expr))

        node = res.register(self.comp_expr())
        if res.error: return res

        while self.current_tok.matches(TT_KEYWORD, 'ДА') or self.current_tok.matches(TT_KEYWORD, 'ЛИБО'):
            op_tok = self.current_tok
            res.register_advancement()
            self.advance()
            right = res.register(self.comp_expr())
            if res.error: return res
            node = BinOpNode(node, op_tok, right)

        return res.success(node)

    def comp_expr(self):
        res = ParseResult()

        if self.current_tok.matches(TT_KEYWORD, 'НЕ'):
            op_tok = self.current_tok
            res.register_advancement()
            self.advance()
            node = res.register(self.comp_expr())
            if res.error: return res
            return res.success(UnaryOpNode(op_tok, node))

        node = res.register(self.arith_expr())
        if res.error: return res

        while self.current_tok.type in (TT_EE, TT_NE, TT_LT, TT_GT, TT_LTE, TT_GTE):
            op_tok = self.current_tok
            res.register_advancement()
            self.advance()
            right = res.register(self.arith_expr())
            if res.error: return res
            node = BinOpNode(node, op_tok, right)

        return res.success(node)

    def arith_expr(self):
        return self.bin_op(self.term, (TT_PLUS, TT_MINUS))

    def term(self):
        return self.bin_op(self.factor, (TT_MUL, TT_DIV))

    def factor(self):
        res = ParseResult()
        tok = self.current_tok

        if tok.type in (TT_PLUS, TT_MINUS):
            res.register_advancement()
            self.advance()
            factor = res.register(self.factor())
            if res.error: return res
            return res.success(UnaryOpNode(tok, factor))

        return self.power()

    def power(self):
        return self.bin_op(self.call, (TT_POW,), self.factor)

    def call(self):
        res = ParseResult()
        atom = res.register(self.atom())
        if res.error: return res

        if self.current_tok.type == TT_LPAREN:
            res.register_advancement()
            self.advance()
            arg_nodes = []

            if self.current_tok.type == TT_RPAREN:
                res.register_advancement()
                self.advance()
            else:
                arg_nodes.append(res.register(self.expr()))
                if res.error:
                    return res.failure(InvalidSyntaxError(
                        self.current_tok.pos_start, self.current_tok.pos_end,
                        "Ожидалось ')', 'ВЪ', 'ЕСЛИ', 'ПО', 'ПОКА', 'ВЕСЕЛЬЕ', целое, дробное, идентификатор, '[', '(' или 'НЕ'"
                    ))

                while self.current_tok.type == TT_COMMA:
                    res.register_advancement()
                    self.advance()
                    arg_nodes.append(res.register(self.expr()))
                    if res.error: return res

                if self.current_tok.type != TT_RPAREN:
                    return res.failure(InvalidSyntaxError(
                        self.current_tok.pos_start, self.current_tok.pos_end,
                        "Ожидалось ',' или ')'"
                    ))

                res.register_advancement()
                self.advance()
            return res.success(CallNode(atom, arg_nodes))
        return res.success(atom)

    def atom(self):
        res = ParseResult()
        tok = self.current_tok

        if tok.type in (TT_INT, TT_FLOAT):
            res.register_advancement()
            self.advance()
            return res.success(NumberNode(tok))

        elif tok.type == TT_STRING:
            res.register_advancement()
            self.advance()
            return res.success(StringNode(tok))

        elif tok.type == TT_IDENTIFIER:
            res.register_advancement()
            self.advance()
            return res.success(VarAccessNode(tok))

        elif tok.type == TT_LPAREN:
            res.register_advancement()
            self.advance()
            expr = res.register(self.expr())
            if res.error: return res
            if self.current_tok.type == TT_RPAREN:
                res.register_advancement()
                self.advance()
                return res.success(expr)
            else:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Ожидалось ')'"
                ))

        elif tok.type == TT_LSQUARE:
            list_expr = res.register(self.list_expr())
            if res.error: return res
            return res.success(list_expr)

        elif tok.matches(TT_KEYWORD, 'ЕСЛИ'):
            if_expr = res.register(self.if_expr())
            if res.error: return res
            return res.success(if_expr)

        elif tok.matches(TT_KEYWORD, 'ПО'):
            for_expr = res.register(self.for_expr())
            if res.error: return res
            return res.success(for_expr)

        elif tok.matches(TT_KEYWORD, 'ПОКА'):
            while_expr = res.register(self.while_expr())
            if res.error: return res
            return res.success(while_expr)

        elif tok.matches(TT_KEYWORD, 'ВЕСЕЛЬЕ'):
            func_def = res.register(self.func_def())
            if res.error: return res
            return res.success(func_def)

        return res.failure(InvalidSyntaxError(
            tok.pos_start, tok.pos_end,
            "Ожидалось целое, дробное, идентификатор, '[', '(', '+', '-', 'ЕСЛИ', 'ПО', 'ПОКА' или 'ВЕСЕЛЬЕ'"
        ))

    def list_expr(self):
        res = ParseResult()
        element_nodes = []
        pos_start = self.current_tok.pos_start.copy()

        if self.current_tok.type != TT_LSQUARE:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Ожидалось '['"
            ))

        res.register_advancement()
        self.advance()

        if self.current_tok.type == TT_RSQUARE:
            res.register_advancement()
            self.advance()
        else:
            element_nodes.append(res.register(self.expr()))
            if res.error:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Ожидалось ']', 'ВЪ', 'ЕСЛИ', 'ПО', 'ПОКА', 'ВЕСЕЛЬЕ', целое, дробное, идентификатор, '[', '(' или 'НЕ'"
                ))

            while self.current_tok.type == TT_COMMA:
                res.register_advancement()
                self.advance()
                element_nodes.append(res.register(self.expr()))
                if res.error: return res

            if self.current_tok.type != TT_RSQUARE:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Ожидалось ',' или ']'"
                ))

            res.register_advancement()
            self.advance()

        return res.success(ListNode(
            element_nodes, pos_start, self.current_tok.pos_end.copy()
        ))

    def if_expr(self):
        res = ParseResult()
        all_cases = res.register(self.if_expr_cases('ЕСЛИ'))
        if res.error: return res
        cases, else_case = all_cases
        return res.success(IfNode(cases, else_case))

    def if_expr_b(self):
        return self.if_expr_cases('КОЛИ')

    def if_expr_c(self):
        res = ParseResult()
        else_case = None

        if self.current_tok.matches(TT_KEYWORD, 'ИЛИ'):
            res.register_advancement()
            self.advance()

            if self.current_tok.type == TT_NEWLINE:
                res.register_advancement()
                self.advance()

                statements = res.register(self.statements())
                if res.error: return res
                else_case = (statements, True)

                if self.current_tok.matches(TT_KEYWORD, 'ЗАКОНЧИТЬ'):
                    res.register_advancement()
                    self.advance()
                else:
                    return res.failure(InvalidSyntaxError(
                        self.current_tok.pos_start, self.current_tok.pos_end,
                        "Ожидалось 'ЗАКОНЧИТЬ'"
                    ))
            else:
                expr = res.register(self.statement())
                if res.error: return res
                else_case = (expr, False)

        return res.success(else_case)

    def if_expr_b_or_c(self):
        res = ParseResult()
        cases, else_case = [], None

        if self.current_tok.matches(TT_KEYWORD, 'КОЛИ'):
            all_cases = res.register(self.if_expr_b())
            if res.error: return res
            cases, else_case = all_cases
        else:
            else_case = res.register(self.if_expr_c())
            if res.error: return res

        return res.success((cases, else_case))

    def if_expr_cases(self, case_keyword):
        res = ParseResult()
        cases = []
        else_case = None

        if not self.current_tok.matches(TT_KEYWORD, case_keyword):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Ожидалось '{case_keyword}'"
            ))

        res.register_advancement()
        self.advance()

        condition = res.register(self.expr())
        if res.error: return res

        if not self.current_tok.matches(TT_KEYWORD, 'ПОСЛЕ'):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Ожидалось 'ПОСЛЕ'"
            ))

        res.register_advancement()
        self.advance()

        if self.current_tok.type == TT_NEWLINE:
            res.register_advancement()
            self.advance()

            statements = res.register(self.statements())
            if res.error: return res
            cases.append((condition, statements, True))

            if self.current_tok.matches(TT_KEYWORD, 'ЗАКОНЧИТЬ'):
                res.register_advancement()
                self.advance()
            else:
                all_cases = res.register(self.if_expr_b_or_c())
                if res.error: return res
                new_cases, else_case = all_cases
                cases.extend(new_cases)
        else:
            expr = res.register(self.statement())
            if res.error: return res
            cases.append((condition, expr, False))

            all_cases = res.register(self.if_expr_b_or_c())
            if res.error: return res
            new_cases, else_case = all_cases
            cases.extend(new_cases)

        return res.success((cases, else_case))

    def for_expr(self):
        res = ParseResult()

        if not self.current_tok.matches(TT_KEYWORD, 'ПО'):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Ожидалось 'ПО'"
            ))

        res.register_advancement()
        self.advance()

        if self.current_tok.type != TT_IDENTIFIER:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Ожидался идентификатор"
            ))

        var_name = self.current_tok
        res.register_advancement()
        self.advance()

        if self.current_tok.type != TT_EQ:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Ожидалось '='"
            ))

        res.register_advancement()
        self.advance()

        start_value = res.register(self.expr())
        if res.error: return res

        if not self.current_tok.matches(TT_KEYWORD, 'ДО'):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Ожидалось 'ДО'"
            ))

        res.register_advancement()
        self.advance()

        end_value = res.register(self.expr())
        if res.error: return res

        if self.current_tok.matches(TT_KEYWORD, 'ШАГ'):
            res.register_advancement()
            self.advance()

            step_value = res.register(self.expr())
            if res.error: return res
        else:
            step_value = None

        if not self.current_tok.matches(TT_KEYWORD, 'ПОСЛЕ'):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Ожидалось 'ПОСЛЕ'"
            ))

        res.register_advancement()
        self.advance()

        if self.current_tok.type == TT_NEWLINE:
            res.register_advancement()
            self.advance()

            body = res.register(self.statements())
            if res.error: return res

            if not self.current_tok.matches(TT_KEYWORD, 'ЗАКОНЧИТЬ'):
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Ожидалось 'ЗАКОНЧИТЬ'"
                ))

            res.register_advancement()
            self.advance()

            return res.success(ForNode(var_name, start_value, end_value, step_value, body, True))

        body = res.register(self.statement())
        if res.error: return res

        return res.success(ForNode(var_name, start_value, end_value, step_value, body, False))

    def while_expr(self):
        res = ParseResult()

        if not self.current_tok.matches(TT_KEYWORD, 'ПОКА'):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Ожидалось 'ПОКА'"
            ))

        res.register_advancement()
        self.advance()

        condition = res.register(self.expr())
        if res.error: return res

        if not self.current_tok.matches(TT_KEYWORD, 'ПОСЛЕ'):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Ожидалось 'ПОСЛЕ'"
            ))

        res.register_advancement()
        self.advance()

        if self.current_tok.type == TT_NEWLINE:
            res.register_advancement()
            self.advance()

            body = res.register(self.statements())
            if res.error: return res

            if not self.current_tok.matches(TT_KEYWORD, 'ЗАКОНЧИТЬ'):
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Ожидалось 'ЗАКОНЧИТЬ'"
                ))

            res.register_advancement()
            self.advance()

            return res.success(WhileNode(condition, body, True))

        body = res.register(self.statement())
        if res.error: return res

        return res.success(WhileNode(condition, body, False))

    def func_def(self):
        res = ParseResult()

        if not self.current_tok.matches(TT_KEYWORD, 'ВЕСЕЛЬЕ'):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Ожидалось 'ВЕСЕЛЬЕ'"
            ))

        res.register_advancement()
        self.advance()

        if self.current_tok.type == TT_IDENTIFIER:
            var_name_tok = self.current_tok
            res.register_advancement()
            self.advance()
            if self.current_tok.type != TT_LPAREN:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Ожидалось '('"
                ))
        else:
            var_name_tok = None
            if self.current_tok.type != TT_LPAREN:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Ожидалось идентификатор или '('"
                ))

        res.register_advancement()
        self.advance()
        arg_name_toks = []

        if self.current_tok.type == TT_IDENTIFIER:
            arg_name_toks.append(self.current_tok)
            res.register_advancement()
            self.advance()

            while self.current_tok.type == TT_COMMA:
                res.register_advancement()
                self.advance()

                if self.current_tok.type != TT_IDENTIFIER:
                    return res.failure(InvalidSyntaxError(
                        self.current_tok.pos_start, self.current_tok.pos_end,
                        "Ожидался идентификатор"
                    ))

                arg_name_toks.append(self.current_tok)
                res.register_advancement()
                self.advance()

            if self.current_tok.type != TT_RPAREN:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Ожидалось ',' или ')'"
                ))
        else:
            if self.current_tok.type != TT_RPAREN:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Ожидался идентификатор или ')'"
                ))

        res.register_advancement()
        self.advance()

        if self.current_tok.type == TT_ARROW:
            res.register_advancement()
            self.advance()

            body = res.register(self.expr())
            if res.error: return res

            return res.success(FuncDefNode(
                var_name_tok, arg_name_toks, body, True
            ))

        if self.current_tok.type != TT_NEWLINE:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Ожидалось '->' или новая строка"
            ))

        res.register_advancement()
        self.advance()

        body = res.register(self.statements())
        if res.error: return res

        if not self.current_tok.matches(TT_KEYWORD, 'ЗАКОНЧИТЬ'):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Ожидалось 'ЗАКОНЧИТЬ'"
            ))

        res.register_advancement()
        self.advance()

        return res.success(FuncDefNode(
            var_name_tok, arg_name_toks, body, False
        ))

    def bin_op(self, func_a, ops, func_b=None):
        if func_b is None:
            func_b = func_a

        res = ParseResult()
        left = res.register(func_a())
        if res.error: return res

        while self.current_tok.type in ops or (self.current_tok.type, self.current_tok.value) in ops:
            op_tok = self.current_tok
            res.register_advancement()
            self.advance()
            right = res.register(func_b())
            if res.error: return res
            left = BinOpNode(left, op_tok, right)

        return res.success(left)

#######################################
# РЕЗУЛЬТАТ ВЫПОЛНЕНИЯ
#######################################

class RTResult:
    def __init__(self):
        self.value = None
        self.error = None
        self.func_return_value = None
        self.loop_should_continue = False
        self.loop_should_break = False

    def register(self, res):
        if res.error: self.error = res.error
        self.func_return_value = res.func_return_value
        self.loop_should_continue = res.loop_should_continue
        self.loop_should_break = res.loop_should_break
        return res.value

    def success(self, value):
        self.value = value
        return self

    def success_return(self, value):
        self.func_return_value = value
        return self

    def success_continue(self):
        self.loop_should_continue = True
        return self

    def success_break(self):
        self.loop_should_break = True
        return self

    def failure(self, error):
        self.error = error
        return self

    def should_return(self):
        return (
            self.error or
            self.func_return_value or
            self.loop_should_continue or
            self.loop_should_break
        )

#######################################
# ЗНАЧЕНИЯ
#######################################

class Value:
    def __init__(self):
        self.set_pos()
        self.set_context()

    def set_pos(self, pos_start=None, pos_end=None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self

    def set_context(self, context=None):
        self.context = context
        return self

    def added_to(self, other):
        return None, self.illegal_operation(other)

    def subbed_by(self, other):
        return None, self.illegal_operation(other)

    def multed_by(self, other):
        return None, self.illegal_operation(other)

    def dived_by(self, other):
        return None, self.illegal_operation(other)

    def powed_by(self, other):
        return None, self.illegal_operation(other)

    def get_comparison_eq(self, other):
        return None, self.illegal_operation(other)

    def get_comparison_ne(self, other):
        return None, self.illegal_operation(other)

    def get_comparison_lt(self, other):
        return None, self.illegal_operation(other)

    def get_comparison_gt(self, other):
        return None, self.illegal_operation(other)

    def get_comparison_lte(self, other):
        return None, self.illegal_operation(other)

    def get_comparison_gte(self, other):
        return None, self.illegal_operation(other)

    def anded_by(self, other):
        return None, self.illegal_operation(other)

    def ored_by(self, other):
        return None, self.illegal_operation(other)

    def notted(self):
        return None, self.illegal_operation()

    def execute(self, args):
        return RTResult().failure(self.illegal_operation())

    def copy(self):
        raise Exception('Метод copy не определён')

    def is_true(self):
        return False

    def illegal_operation(self, other=None):
        if not other: other = self
        return RTError(
            self.pos_start, other.pos_end,
            'Недопустимая операция', self.context
        )

class Number(Value):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def added_to(self, other):
        if isinstance(other, Number):
            return Number(self.value + other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def subbed_by(self, other):
        if isinstance(other, Number):
            return Number(self.value - other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def multed_by(self, other):
        if isinstance(other, Number):
            return Number(self.value * other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def dived_by(self, other):
        if isinstance(other, Number):
            if other.value == 0:
                return None, RTError(
                    other.pos_start, other.pos_end,
                    'Деление на ноль', self.context
                )
            return Number(self.value / other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def powed_by(self, other):
        if isinstance(other, Number):
            return Number(self.value ** other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def get_comparison_eq(self, other):
        if isinstance(other, Number):
            return Number(int(self.value == other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def get_comparison_ne(self, other):
        if isinstance(other, Number):
            return Number(int(self.value != other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def get_comparison_lt(self, other):
        if isinstance(other, Number):
            return Number(int(self.value < other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def get_comparison_gt(self, other):
        if isinstance(other, Number):
            return Number(int(self.value > other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def get_comparison_lte(self, other):
        if isinstance(other, Number):
            return Number(int(self.value <= other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def get_comparison_gte(self, other):
        if isinstance(other, Number):
            return Number(int(self.value >= other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def anded_by(self, other):
        if isinstance(other, Number):
            return Number(int(self.value and other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def ored_by(self, other):
        if isinstance(other, Number):
            return Number(int(self.value or other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def notted(self):
        return Number(1 if self.value == 0 else 0).set_context(self.context), None

    def copy(self):
        copy = Number(self.value)
        copy.set_pos(self.pos_start, self.pos_end)
        copy.set_context(self.context)
        return copy

    def is_true(self):
        return self.value != 0

    def __repr__(self):
        return str(self.value)

Number.null = Number(0)
Number.false = Number(0)
Number.true = Number(1)
Number.math_PI = Number(math.pi)

class String(Value):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def added_to(self, other):
        if isinstance(other, String):
            return String(self.value + other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def multed_by(self, other):
        if isinstance(other, Number):
            return String(self.value * other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def is_true(self):
        return len(self.value) > 0

    def copy(self):
        copy = String(self.value)
        copy.set_pos(self.pos_start, self.pos_end)
        copy.set_context(self.context)
        return copy

    def __repr__(self):
        return f'"{self.value}"'

class List(Value):
    def __init__(self, elements):
        super().__init__()
        self.elements = elements

    def added_to(self, other):
        new_list = self.copy()
        new_list.elements.append(other)
        return new_list, None

    def copy(self):
        copy = List(self.elements[:])
        copy.set_pos(self.pos_start, self.pos_end)
        copy.set_context(self.context)
        return copy

    def __repr__(self):
        return f'[{", ".join([str(x) for x in self.elements])}]'

class BaseFunction(Value):
    def __init__(self, name):
        super().__init__()
        self.name = name or "<анонимная функция>"

    def generate_new_context(self):
        new_context = Context(self.name, self.context, self.pos_start)
        return new_context

    def check_args(self, arg_names, args):
        res = RTResult()

        if len(args) > len(arg_names):
            return res.failure(RTError(
                self.pos_start, self.pos_end,
                f"Передано слишком много аргументов ({len(args) - len(arg_names)}) в {self}",
                self.context
            ))

        if len(args) < len(arg_names):
            return res.failure(RTError(
                self.pos_start, self.pos_end,
                f"Передано слишком мало аргументов ({len(arg_names) - len(args)}) в {self}",
                self.context
            ))

        return res.success(None)

    def populate_args(self, arg_names, args, exec_ctx):
        for i in range(len(args)):
            arg_name = arg_names[i]
            arg_value = args[i]
            arg_value.set_context(exec_ctx)
            exec_ctx.symbol_table.set(arg_name, arg_value)

    def check_and_populate_args(self, arg_names, args, exec_ctx):
        res = RTResult()
        res.register(self.check_args(arg_names, args))
        if res.error: return res
        self.populate_args(arg_names, args, exec_ctx)
        return res.success(None)

class Function(BaseFunction):
    def __init__(self, name, body_node, arg_names, should_auto_return):
        super().__init__(name)
        self.body_node = body_node
        self.arg_names = arg_names
        self.should_auto_return = should_auto_return

    def execute(self, args):
        res = RTResult()
        interpreter = Interpreter()
        exec_ctx = self.generate_new_context()

        res.register(self.check_and_populate_args(self.arg_names, args, exec_ctx))
        if res.error: return res

        value = res.register(interpreter.visit(self.body_node, exec_ctx))
        if res.should_return() and res.func_return_value: return res
        ret_value = (value if self.should_auto_return else None) or res.func_return_value or Number.null
        return res.success(ret_value)

    def copy(self):
        copy = Function(self.name, self.body_node, self.arg_names, self.should_auto_return)
        copy.set_context(self.context)
        copy.set_pos(self.pos_start, self.pos_end)
        return copy

    def __repr__(self):
        return f"<функция {self.name}>"

class BuiltInFunction(BaseFunction):
    def __init__(self, name):
        super().__init__(name)

    def execute(self, args):
        res = RTResult()
        method_name = f"execute_{self.name}"
        method = getattr(self, method_name, self.no_execute)
        return method(args)

    def no_execute(self, args):
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            f"Функция '{self.name}' не реализована",
            self.context
        ))

    def execute_ПИШИ(self, args):
        res = RTResult()
        if len(args) < 1:
            return res.failure(RTError(
                self.pos_start, self.pos_end,
                "Ожидается хотя бы один аргумент для функции ПИШИ",
                self.context
            ))
        print(str(args[0]))
        return res.success(Number.null)

    def execute_ПЕЧАТЬ_ВЕРНИ(self, args):
        res = RTResult()
        if len(args) < 1:
            return res.failure(RTError(
                self.pos_start, self.pos_end,
                "Ожидается хотя бы один аргумент для функции ПЕЧАТЬ_ВЕРНИ",
                self.context
            ))
        return res.success(String(str(args[0])))

    def execute_ВВОД(self, args):
        res = RTResult()
        try:
            user_input = input()
            return res.success(String(user_input))
        except Exception as e:
            return res.failure(RTError(
                self.pos_start, self.pos_end,
                f"Ошибка при выполнении ВВОД: {str(e)}",
                self.context
            ))

    def execute_ВВОД_ЦЕЛОЕ(self, args):
        res = RTResult()
        try:
            user_input = input()
            return res.success(Number(int(user_input)))
        except ValueError:
            return res.failure(RTError(
                self.pos_start, self.pos_end,
                "Ожидалось целое число",
                self.context
            ))
        except Exception as e:
            return res.failure(RTError(
                self.pos_start, self.pos_end,
                f"Ошибка при выполнении ВВОД_ЦЕЛОЕ: {str(e)}",
                self.context
            ))

    def execute_ОЧИСТИТЬ(self, args):
        res = RTResult()
        os.system('cls' if os.name == 'nt' else 'clear')
        return res.success(Number.null)

    def execute_ЭТО_ЧИСЛО(self, args):
        res = RTResult()
        if len(args) != 1:
            return res.failure(RTError(
                self.pos_start, self.pos_end,
                "Ожидается ровно один аргумент",
                self.context
            ))
        return res.success(Number.true if isinstance(args[0], Number) else Number.false)

    def execute_ЭТО_СТРОКА(self, args):
        res = RTResult()
        if len(args) != 1:
            return res.failure(RTError(
                self.pos_start, self.pos_end,
                "Ожидается ровно один аргумент",
                self.context
            ))
        return res.success(Number.true if isinstance(args[0], String) else Number.false)

    def execute_ЭТО_СПИСОК(self, args):
        res = RTResult()
        if len(args) != 1:
            return res.failure(RTError(
                self.pos_start, self.pos_end,
                "Ожидается ровно один аргумент",
                self.context
            ))
        return res.success(Number.true if isinstance(args[0], List) else Number.false)

    def execute_ЭТО_ФУНКЦИЯ(self, args):
        res = RTResult()
        if len(args) != 1:
            return res.failure(RTError(
                self.pos_start, self.pos_end,
                "Ожидается ровно один аргумент",
                self.context
            ))
        return res.success(Number.true if isinstance(args[0], BaseFunction) else Number.false)

    def execute_ДОБАВИТЬ(self, args):
        res = RTResult()
        if len(args) != 2:
            return res.failure(RTError(
                self.pos_start, self.pos_end,
                "Ожидается ровно два аргумента",
                self.context
            ))
        if not isinstance(args[0], List):
            return res.failure(RTError(
                self.pos_start, self.pos_end,
                "Первый аргумент должен быть списком",
                self.context
            ))
        new_list = args[0].copy()
        new_list.elements.append(args[1])
        return res.success(new_list)

    def execute_ИЗВЛЕЧЬ(self, args):
        res = RTResult()
        if len(args) != 2:
            return res.failure(RTError(
                self.pos_start, self.pos_end,
                "Ожидается ровно два аргумента",
                self.context
            ))
        if not isinstance(args[0], List):
            return res.failure(RTError(
                self.pos_start, self.pos_end,
                "Первый аргумент должен быть списком",
                self.context
            ))
        if not isinstance(args[1], Number):
            return res.failure(RTError(
                self.pos_start, self.pos_end,
                "Второй аргумент должен быть числом",
                self.context
            ))
        try:
            index = args[1].value
            element = args[0].elements[index]
            return res.success(element)
        except IndexError:
            return res.failure(RTError(
                self.pos_start, self.pos_end,
                "Индекс вне диапазона",
                self.context
            ))

    def execute_РАСШИРИТЬ(self, args):
        res = RTResult()
        if len(args) != 2:
            return res.failure(RTError(
                self.pos_start, self.pos_end,
                "Ожидается ровно два аргумента",
                self.context
            ))
        if not isinstance(args[0], List):
            return res.failure(RTError(
                self.pos_start, self.pos_end,
                "Первый аргумент должен быть списком",
                self.context
            ))
        if not isinstance(args[1], List):
            return res.failure(RTError(
                self.pos_start, self.pos_end,
                "Второй аргумент должен быть списком",
                self.context
            ))
        new_list = args[0].copy()
        new_list.elements.extend(args[1].elements)
        return res.success(new_list)

    def execute_ДЛИНА(self, args):
        res = RTResult()
        if len(args) != 1:
            return res.failure(RTError(
                self.pos_start, self.pos_end,
                "Ожидается ровно один аргумент",
                self.context
            ))
        if not isinstance(args[0], List):
            return res.failure(RTError(
                self.pos_start, self.pos_end,
                "Аргумент должен быть списком",
                self.context
            ))
        return res.success(Number(len(args[0].elements)))

    def execute_ЗАПУСТИТЬ(self, args):
        res = RTResult()
        if len(args) != 1:
            return res.failure(RTError(
                self.pos_start, self.pos_end,
                "Ожидается ровно один аргумент",
                self.context
            ))
        if not isinstance(args[0], String):
            return res.failure(RTError(
                self.pos_start, self.pos_end,
                "Аргумент должен быть строкой",
                self.context
            ))
        try:
            with open(args[0].value, 'r', encoding='utf-8') as f:
                text = f.read()
            lexer = Lexer(args[0].value, text)
            tokens, error = lexer.make_tokens()
            if error: return res.failure(error)

            parser = Parser(tokens)
            ast = parser.parse()
            if ast.error: return res.failure(ast.error)

            interpreter = Interpreter()
            new_context = Context('<программа>')
            new_context.symbol_table = SymbolTable()
            result = interpreter.visit(ast.node, new_context)
            if result.error: return res.failure(result.error)
            return res.success(result.value)
        except Exception as e:
            return res.failure(RTError(
                self.pos_start, self.pos_end,
                f"Ошибка при выполнении файла: {str(e)}",
                self.context
            ))

    def copy(self):
        copy = BuiltInFunction(self.name)
        copy.set_context(self.context)
        copy.set_pos(self.pos_start, self.pos_end)
        return copy

    def __repr__(self):
        return f"<встроенная функция {self.name}>"

# Определение встроенных функций
BuiltInFunction.print = BuiltInFunction("ПИШИ")
BuiltInFunction.print_ret = BuiltInFunction("ПЕЧАТЬ_ВЕРНИ")
BuiltInFunction.input = BuiltInFunction("ВВОД")
BuiltInFunction.input_int = BuiltInFunction("ВВОД_ЦЕЛОЕ")
BuiltInFunction.clear = BuiltInFunction("ОЧИСТИТЬ")
BuiltInFunction.is_number = BuiltInFunction("ЭТО_ЧИСЛО")
BuiltInFunction.is_string = BuiltInFunction("ЭТО_СТРОКА")
BuiltInFunction.is_list = BuiltInFunction("ЭТО_СПИСОК")
BuiltInFunction.is_function = BuiltInFunction("ЭТО_ФУНКЦИЯ")
BuiltInFunction.append = BuiltInFunction("ДОБАВИТЬ")
BuiltInFunction.pop = BuiltInFunction("ИЗВЛЕЧЬ")
BuiltInFunction.extend = BuiltInFunction("РАСШИРИТЬ")
BuiltInFunction.len = BuiltInFunction("ДЛИНА")
BuiltInFunction.run = BuiltInFunction("ЗАПУСТИТЬ")

#######################################
# ТАБЛИЦА СИМВОЛОВ
#######################################

class SymbolTable:
    def __init__(self, parent=None):
        self.symbols = {}
        self.parent = parent

    def get(self, name):
        value = self.symbols.get(name, None)
        if value is None and self.parent:
            return self.parent.get(name)
        return value

    def set(self, name, value):
        self.symbols[name] = value

    def remove(self, name):
        del self.symbols[name]

#######################################
# КОНТЕКСТ
#######################################

class Context:
    def __init__(self, display_name, parent=None, parent_entry_pos=None):
        self.display_name = display_name
        self.parent = parent
        self.parent_entry_pos = parent_entry_pos
        self.symbol_table = None

#######################################
# ИНТЕРПРЕТАТОР
#######################################

class Interpreter:
    def visit(self, node, context):
        method_name = f'visit_{type(node).__name__}'
        method = getattr(self, method_name, self.no_visit_method)
        return method(node, context)

    def no_visit_method(self, node, context):
        raise Exception(f'Метод visit_{type(node).__name__} не определён')

    def visit_NumberNode(self, node, context):
        return RTResult().success(
            Number(node.tok.value).set_context(context).set_pos(node.pos_start, node.pos_end)
        )

    def visit_StringNode(self, node, context):
        return RTResult().success(
            String(node.tok.value).set_context(context).set_pos(node.pos_start, node.pos_end)
        )

    def visit_ListNode(self, node, context):
        res = RTResult()
        elements = []
        for element_node in node.element_nodes:
            elements.append(res.register(self.visit(element_node, context)))
            if res.error: return res
        return res.success(List(elements).set_context(context).set_pos(node.pos_start, node.pos_end))

    def visit_VarAccessNode(self, node, context):
        res = RTResult()
        var_name = node.var_name_tok.value
        value = context.symbol_table.get(var_name)

        if value is None:
            return res.failure(RTError(
                node.pos_start, node.pos_end,
                f"'{var_name}' не определена",
                context
            ))

        return res.success(value.copy().set_pos(node.pos_start, node.pos_end))

    def visit_VarAssignNode(self, node, context):
        res = RTResult()
        var_name = node.var_name_tok.value
        value = res.register(self.visit(node.value_node, context))
        if res.should_return(): return res

        context.symbol_table.set(var_name, value)
        return res.success(value)

    def visit_BinOpNode(self, node, context):
        res = RTResult()
        left = res.register(self.visit(node.left_node, context))
        if res.should_return(): return res
        right = res.register(self.visit(node.right_node, context))
        if res.should_return(): return res

        if node.op_tok.type == TT_PLUS:
            result, error = left.added_to(right)
        elif node.op_tok.type == TT_MINUS:
            result, error = left.subbed_by(right)
        elif node.op_tok.type == TT_MUL:
            result, error = left.multed_by(right)
        elif node.op_tok.type == TT_DIV:
            result, error = left.dived_by(right)
        elif node.op_tok.type == TT_POW:
            result, error = left.powed_by(right)
        elif node.op_tok.type == TT_EE:
            result, error = left.get_comparison_eq(right)
        elif node.op_tok.type == TT_NE:
            result, error = left.get_comparison_ne(right)
        elif node.op_tok.type == TT_LT:
            result, error = left.get_comparison_lt(right)
        elif node.op_tok.type == TT_GT:
            result, error = left.get_comparison_gt(right)
        elif node.op_tok.type == TT_LTE:
            result, error = left.get_comparison_lte(right)
        elif node.op_tok.type == TT_GTE:
            result, error = left.get_comparison_gte(right)
        elif node.op_tok.matches(TT_KEYWORD, 'ДА'):
            result, error = left.anded_by(right)
        elif node.op_tok.matches(TT_KEYWORD, 'ЛИБО'):
            result, error = left.ored_by(right)

        if error:
            return res.failure(error)
        else:
            return res.success(result.set_pos(node.pos_start, node.pos_end))

    def visit_UnaryOpNode(self, node, context):
        res = RTResult()
        number = res.register(self.visit(node.node, context))
        if res.should_return(): return res

        error = None

        if node.op_tok.type == TT_MINUS:
            number, error = number.multed_by(Number(-1))
        elif node.op_tok.matches(TT_KEYWORD, 'НЕ'):
            number, error = number.notted()

        if error:
            return res.failure(error)
        else:
            return res.success(number.set_pos(node.pos_start, node.pos_end))

    def visit_IfNode(self, node, context):
        res = RTResult()

        for condition, expr, should_return_null in node.cases:
            condition_value = res.register(self.visit(condition, context))
            if res.should_return(): return res

            if condition_value.is_true():
                expr_value = res.register(self.visit(expr, context))
                if res.should_return(): return res
                return res.success(Number.null if should_return_null else expr_value)

        if node.else_case:
            expr, should_return_null = node.else_case
            expr_value = res.register(self.visit(expr, context))
            if res.should_return(): return res
            return res.success(Number.null if should_return_null else expr_value)

        return res.success(Number.null)

    def visit_ForNode(self, node, context):
        res = RTResult()
        elements = []

        start_value = res.register(self.visit(node.start_value_node, context))
        if res.should_return(): return res

        end_value = res.register(self.visit(node.end_value_node, context))
        if res.should_return(): return res

        if node.step_value_node:
            step_value = res.register(self.visit(node.step_value_node, context))
            if res.should_return(): return res
        else:
            step_value = Number(1)

        i = start_value.value

        if step_value.value >= 0:
            condition = lambda: i < end_value.value
        else:
            condition = lambda: i > end_value.value

        while condition():
            context.symbol_table.set(node.var_name_tok.value, Number(i))
            i += step_value.value

            value = res.register(self.visit(node.body_node, context))
            if res.should_return() and not res.loop_should_continue and not res.loop_should_break: return res

            if res.loop_should_continue:
                continue

            if res.loop_should_break:
                break

            elements.append(value)

        return res.success(
            Number.null if node.should_return_null else
            List(elements).set_context(context).set_pos(node.pos_start, node.pos_end)
        )

    def visit_WhileNode(self, node, context):
        res = RTResult()
        elements = []

        while True:
            condition = res.register(self.visit(node.condition_node, context))
            if res.should_return(): return res

            if not condition.is_true():
                break

            value = res.register(self.visit(node.body_node, context))
            if res.should_return() and not res.loop_should_continue and not res.loop_should_break: return res

            if res.loop_should_continue:
                continue

            if res.loop_should_break:
                break

            elements.append(value)

        return res.success(
            Number.null if node.should_return_null else
            List(elements).set_context(context).set_pos(node.pos_start, node.pos_end)
        )

    def visit_FuncDefNode(self, node, context):
        res = RTResult()

        func_name = node.var_name_tok.value if node.var_name_tok else None
        body_node = node.body_node
        arg_names = [arg_name.value for arg_name in node.arg_name_toks]
        func_value = Function(func_name, body_node, arg_names, node.should_auto_return).set_context(context).set_pos(node.pos_start, node.pos_end)

        if node.var_name_tok:
            context.symbol_table.set(func_name, func_value)

        return res.success(func_value)

    def visit_CallNode(self, node, context):
        res = RTResult()
        args = []

        value_to_call = res.register(self.visit(node.node_to_call, context))
        if res.should_return(): return res
        value_to_call = value_to_call.copy().set_pos(node.pos_start, node.pos_end)

        for arg_node in node.arg_nodes:
            args.append(res.register(self.visit(arg_node, context)))
            if res.should_return(): return res

        return_value = res.register(value_to_call.execute(args))
        if res.should_return(): return res
        return_value = return_value.copy().set_pos(node.pos_start, node.pos_end).set_context(context)
        return res.success(return_value)

    def visit_ReturnNode(self, node, context):
        res = RTResult()

        if node.node_to_return:
            value = res.register(self.visit(node.node_to_return, context))
            if res.should_return(): return res
        else:
            value = Number.null

        return res.success_return(value)

    def visit_ContinueNode(self, node, context):
        return RTResult().success_continue()

    def visit_BreakNode(self, node, context):
        return RTResult().success_break()

#######################################
# ТАБЛИЦА СИМВОЛОВ ГЛОБАЛЬНАЯ
#######################################

global_symbol_table = SymbolTable()
global_symbol_table.set("НУЛЬ", Number.null)
global_symbol_table.set("ЛОЖЬ", Number.false)
global_symbol_table.set("ПРАВДА", Number.true)
global_symbol_table.set("ЧИСЛО_ПИ", Number.math_PI)
global_symbol_table.set("ПИШИ", BuiltInFunction.print)
global_symbol_table.set("ПЕЧАТЬ_ВЕРНИ", BuiltInFunction.print_ret)
global_symbol_table.set("ВВОД", BuiltInFunction.input)
global_symbol_table.set("ВВОД_ЦЕЛОЕ", BuiltInFunction.input_int)
global_symbol_table.set("ОЧИСТИТЬ", BuiltInFunction.clear)
global_symbol_table.set("ЭТО_ЧИСЛО", BuiltInFunction.is_number)
global_symbol_table.set("ЭТО_СТРОКА", BuiltInFunction.is_string)
global_symbol_table.set("ЭТО_СПИСОК", BuiltInFunction.is_list)
global_symbol_table.set("ЭТО_ФУНКЦИЯ", BuiltInFunction.is_function)
global_symbol_table.set("ДОБАВИТЬ", BuiltInFunction.append)
global_symbol_table.set("ИЗВЛЕЧЬ", BuiltInFunction.pop)
global_symbol_table.set("РАСШИРИТЬ", BuiltInFunction.extend)
global_symbol_table.set("ДЛИНА", BuiltInFunction.len)
global_symbol_table.set("ЗАПУСТИТЬ", BuiltInFunction.run)

#######################################
# ЗАПУСК
#######################################

def run(fn, text):
    try:
        # Генерация токенов
        lexer = Lexer(fn, text)
        tokens, error = lexer.make_tokens()
        if error: return None, error

        # Генерация AST
        parser = Parser(tokens)
        ast = parser.parse()
        if ast.error: return None, ast.error

        # Выполнение программы
        interpreter = Interpreter()
        context = Context('<программа>')
        context.symbol_table = global_symbol_table
        result = interpreter.visit(ast.node, context)

        return result.value, result.error
    except Exception as e:
        return None, RTError(
            Position(0, 0, 0, fn, text),
            Position(0, 0, 0, fn, text),
            f"Неожиданная ошибка: {str(e)}",
            Context('<программа>')
        )