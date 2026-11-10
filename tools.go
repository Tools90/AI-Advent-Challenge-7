package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
	"time"
	_ "time/tzdata"
)

type ToolSchema struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

type ToolFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
}

type Tool struct {
	Schema ToolSchema
	Run    func(ctx context.Context, args json.RawMessage) (any, error)
}

type TimeResult struct {
	Timezone string `json:"timezone"`
	RFC3339  string `json:"rfc3339"`
	Human    string `json:"human"`
}

type CalcResult struct {
	Expression string  `json:"expression"`
	Result     float64 `json:"result"`
}

type CurrencyResult struct {
	Amount float64 `json:"amount"`
	From   string  `json:"from"`
	To     string  `json:"to"`
	Rate   float64 `json:"rate"`
	Result float64 `json:"result"`
}

var currencyRates = map[string]float64{
	"RUB": 1,
	"USD": 92.5,
	"EUR": 100,
	"CNY": 12.8,
}

func buildTools() map[string]Tool {
	tools := map[string]Tool{}

	tools["get_current_time"] = Tool{
		Schema: ToolSchema{
			Type: "function",
			Function: ToolFunction{
				Name:        "get_current_time",
				Description: "Возвращает текущие дату и время. timezone — IANA-часовой пояс, например Europe/Moscow; по умолчанию UTC.",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"timezone": map[string]any{
							"type":        "string",
							"description": "IANA timezone, например Europe/Moscow. По умолчанию UTC.",
						},
					},
					"additionalProperties": false,
				},
			},
		},
		Run: runTimeTool,
	}

	tools["calculate"] = Tool{
		Schema: ToolSchema{
			Type: "function",
			Function: ToolFunction{
				Name:        "calculate",
				Description: "Считает арифметическое выражение с числами, операциями + - * / и скобками. Проценты передавай десятичной дробью: 15% = 0.15.",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"expression": map[string]any{
							"type":        "string",
							"description": "Арифметическое выражение, например 0.15*200 или (17*23)+5.",
						},
					},
					"required":             []string{"expression"},
					"additionalProperties": false,
				},
			},
		},
		Run: runCalcTool,
	}

	tools["convert_currency"] = Tool{
		Schema: ToolSchema{
			Type: "function",
			Function: ToolFunction{
				Name:        "convert_currency",
				Description: "Конвертирует сумму между RUB, USD, EUR и CNY по зашитому курсу.",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"amount": map[string]any{"type": "number", "description": "Сумма"},
						"from": map[string]any{
							"type": "string",
							"enum": []string{"RUB", "USD", "EUR", "CNY"},
						},
						"to": map[string]any{
							"type": "string",
							"enum": []string{"RUB", "USD", "EUR", "CNY"},
						},
					},
					"required":             []string{"amount", "from", "to"},
					"additionalProperties": false,
				},
			},
		},
		Run: runCurrencyTool,
	}

	return tools
}

func runTimeTool(ctx context.Context, args json.RawMessage) (any, error) {
	var input struct {
		Timezone string `json:"timezone"`
	}
	if len(args) > 0 {
		if err := json.Unmarshal(args, &input); err != nil {
			return nil, fmt.Errorf("аргументы: %w", err)
		}
	}

	timezone := strings.TrimSpace(input.Timezone)
	if timezone == "" {
		timezone = "UTC"
	}
	location, err := time.LoadLocation(timezone)
	if err != nil {
		return nil, fmt.Errorf("неизвестный часовой пояс %q", timezone)
	}

	now := time.Now().In(location)
	return TimeResult{
		Timezone: timezone,
		RFC3339:  now.Format(time.RFC3339),
		Human:    now.Format("02.01.2006 15:04:05"),
	}, nil
}

func runCalcTool(ctx context.Context, args json.RawMessage) (any, error) {
	var input struct {
		Expression string `json:"expression"`
	}
	if err := json.Unmarshal(args, &input); err != nil {
		return nil, fmt.Errorf("аргументы: %w", err)
	}

	expression := strings.TrimSpace(input.Expression)
	if expression == "" {
		return nil, errors.New("пустое выражение")
	}

	parser := &exprParser{s: expression}
	value, err := parser.parseExpr()
	if err != nil {
		return nil, err
	}
	parser.skipSpace()
	if parser.pos != len(parser.s) {
		return nil, fmt.Errorf("неожиданный символ %q", parser.s[parser.pos])
	}
	if math.IsInf(value, 0) || math.IsNaN(value) {
		return nil, errors.New("результат не является конечным числом")
	}

	return CalcResult{Expression: expression, Result: value}, nil
}

func runCurrencyTool(ctx context.Context, args json.RawMessage) (any, error) {
	var input struct {
		Amount float64 `json:"amount"`
		From   string  `json:"from"`
		To     string  `json:"to"`
	}
	if err := json.Unmarshal(args, &input); err != nil {
		return nil, fmt.Errorf("аргументы: %w", err)
	}

	from := strings.ToUpper(strings.TrimSpace(input.From))
	to := strings.ToUpper(strings.TrimSpace(input.To))
	fromRate, ok := currencyRates[from]
	if !ok {
		return nil, fmt.Errorf("неизвестная валюта %q", input.From)
	}
	toRate, ok := currencyRates[to]
	if !ok {
		return nil, fmt.Errorf("неизвестная валюта %q", input.To)
	}

	rate := fromRate / toRate
	result := math.Round(input.Amount*rate*100) / 100
	return CurrencyResult{
		Amount: input.Amount,
		From:   from,
		To:     to,
		Rate:   rate,
		Result: result,
	}, nil
}

type exprParser struct {
	s   string
	pos int
}

func (p *exprParser) skipSpace() {
	for p.pos < len(p.s) && (p.s[p.pos] == ' ' || p.s[p.pos] == '\t' || p.s[p.pos] == '\n' || p.s[p.pos] == '\r') {
		p.pos++
	}
}

func (p *exprParser) parseExpr() (float64, error) {
	value, err := p.parseTerm()
	if err != nil {
		return 0, err
	}

	for {
		p.skipSpace()
		if p.pos >= len(p.s) || (p.s[p.pos] != '+' && p.s[p.pos] != '-') {
			return value, nil
		}
		op := p.s[p.pos]
		p.pos++
		right, err := p.parseTerm()
		if err != nil {
			return 0, err
		}
		if op == '+' {
			value += right
		} else {
			value -= right
		}
	}
}

func (p *exprParser) parseTerm() (float64, error) {
	value, err := p.parseFactor()
	if err != nil {
		return 0, err
	}

	for {
		p.skipSpace()
		if p.pos >= len(p.s) || (p.s[p.pos] != '*' && p.s[p.pos] != '/') {
			return value, nil
		}
		op := p.s[p.pos]
		p.pos++
		right, err := p.parseFactor()
		if err != nil {
			return 0, err
		}
		if op == '*' {
			value *= right
		} else {
			if right == 0 {
				return 0, errors.New("деление на ноль")
			}
			value /= right
		}
	}
}

func (p *exprParser) parseFactor() (float64, error) {
	p.skipSpace()
	if p.pos >= len(p.s) {
		return 0, errors.New("неожиданный конец выражения")
	}

	if p.s[p.pos] == '+' {
		p.pos++
		return p.parseFactor()
	}
	if p.s[p.pos] == '-' {
		p.pos++
		value, err := p.parseFactor()
		return -value, err
	}
	if p.s[p.pos] == '(' {
		p.pos++
		value, err := p.parseExpr()
		if err != nil {
			return 0, err
		}
		p.skipSpace()
		if p.pos >= len(p.s) || p.s[p.pos] != ')' {
			return 0, errors.New("ожидалась закрывающая скобка")
		}
		p.pos++
		return value, nil
	}

	start := p.pos
	for p.pos < len(p.s) {
		ch := p.s[p.pos]
		if (ch >= '0' && ch <= '9') || ch == '.' {
			p.pos++
			continue
		}
		break
	}
	if start == p.pos {
		return 0, fmt.Errorf("ожидалось число в позиции %d", p.pos)
	}

	value, err := strconv.ParseFloat(p.s[start:p.pos], 64)
	if err != nil {
		return 0, fmt.Errorf("некорректное число %q", p.s[start:p.pos])
	}
	return value, nil
}
