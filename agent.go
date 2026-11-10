package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"
)

type TraceStep struct {
	Type      string `json:"type"`
	Tool      string `json:"tool,omitempty"`
	Arguments string `json:"arguments,omitempty"`
	Result    string `json:"result,omitempty"`
	Request   string `json:"request,omitempty"`
	Response  string `json:"response,omitempty"`
	Error     string `json:"error,omitempty"`
	LatencyMs int64  `json:"latencyMs"`
}

type RunResult struct {
	Reply     string
	Messages  []LLMMessage
	Steps     []TraceStep
	Usage     Usage
	LatencyMs int64
	Calls     int
}

type Agent struct {
	llm      LLMClient
	tools    map[string]Tool
	maxSteps int
}

func NewAgent(llm LLMClient, tools map[string]Tool, maxSteps int) *Agent {
	return &Agent{llm: llm, tools: tools, maxSteps: maxSteps}
}

func (a *Agent) Tools() []ToolSchema {
	names := make([]string, 0, len(a.tools))
	for name := range a.tools {
		names = append(names, name)
	}
	sort.Strings(names)

	out := make([]ToolSchema, 0, len(names))
	for _, name := range names {
		out = append(out, a.tools[name].Schema)
	}
	return out
}

func (a *Agent) Run(ctx context.Context, history []LLMMessage, userMessage string) (RunResult, error) {
	start := time.Now()

	system := "Ты — агент с инструментами. Если для ответа нужны время, вычисления или конвертация валюты, вызови подходящий инструмент. " +
		"Не выдумывай результаты инструментов. После получения результатов дай финальный ответ пользователю по-русски."

	messages := make([]LLMMessage, 0, len(history)+2)
	messages = append(messages, LLMMessage{Role: "system", Content: system})
	messages = append(messages, history...)
	messages = append(messages, LLMMessage{Role: "user", Content: userMessage})

	turnStart := 1 + len(history)
	newMessages := func() []LLMMessage {
		return append([]LLMMessage(nil), messages[turnStart:]...)
	}

	schemas := a.Tools()
	var result RunResult

	for step := 0; step < a.maxSteps; step++ {
		callStart := time.Now()
		requestTrace := traceRequest(a.llm.Model(), messages, schemas)
		response, err := a.llm.Chat(ctx, messages, schemas)
		llmTrace := TraceStep{
			Type:      "llm",
			Request:   requestTrace,
			LatencyMs: time.Since(callStart).Milliseconds(),
		}

		result.Calls++
		result.Usage.Add(response.Usage)
		if err != nil {
			llmTrace.Error = err.Error()
			result.Steps = append(result.Steps, llmTrace)
			result.LatencyMs = time.Since(start).Milliseconds()
			result.Messages = newMessages()
			return result, err
		}

		llmTrace.Response = traceResponse(response)
		result.Steps = append(result.Steps, llmTrace)
		messages = append(messages, response.Message)

		if len(response.Message.ToolCalls) == 0 {
			result.Reply = strings.TrimSpace(response.Message.Content)
			result.LatencyMs = time.Since(start).Milliseconds()
			result.Messages = newMessages()
			if result.Reply == "" {
				return result, errors.New("модель вернула пустой ответ")
			}
			return result, nil
		}

		for _, call := range response.Message.ToolCalls {
			stepStart := time.Now()
			trace := TraceStep{
				Type:      "tool",
				Tool:      call.Function.Name,
				Arguments: call.Function.Arguments,
				LatencyMs: time.Since(stepStart).Milliseconds(),
			}

			content := ""
			tool, ok := a.tools[call.Function.Name]
			if !ok {
				trace.Error = fmt.Sprintf("неизвестный инструмент %q", call.Function.Name)
				content = toolErrorContent(trace.Error)
			} else {
				value, err := tool.Run(ctx, json.RawMessage(call.Function.Arguments))
				if err != nil {
					trace.Error = err.Error()
					content = toolErrorContent(err.Error())
				} else {
					encoded, err := json.Marshal(value)
					if err != nil {
						trace.Error = err.Error()
						content = toolErrorContent(err.Error())
					} else {
						trace.Result = string(encoded)
						content = toolOKContent(encoded)
					}
				}
			}

			trace.LatencyMs = time.Since(stepStart).Milliseconds()
			result.Steps = append(result.Steps, trace)
			messages = append(messages, LLMMessage{
				Role:       "tool",
				ToolCallID: call.ID,
				Content:    content,
			})
		}
	}

	result.LatencyMs = time.Since(start).Milliseconds()
	result.Messages = newMessages()
	return result, fmt.Errorf("агент превысил лимит шагов (%d)", a.maxSteps)
}

func toolOKContent(result json.RawMessage) string {
	return `{"ok":true,"result":` + string(result) + `}`
}

func toolErrorContent(message string) string {
	encoded, _ := json.Marshal(map[string]any{"ok": false, "error": message})
	return string(encoded)
}

func traceRequest(model string, messages []LLMMessage, tools []ToolSchema) string {
	payload := map[string]any{
		"model":       model,
		"messages":    messages,
		"tools":       tools,
		"tool_choice": "auto",
		"temperature": 0,
		"thinking":    map[string]string{"type": "disabled"},
	}
	encoded, err := json.Marshal(payload)
	if err != nil {
		return err.Error()
	}
	return string(encoded)
}

func traceResponse(response LLMResponse) string {
	payload := map[string]any{
		"message":       response.Message,
		"finish_reason": response.FinishReason,
		"usage":         response.Usage,
	}
	encoded, err := json.Marshal(payload)
	if err != nil {
		return err.Error()
	}
	return string(encoded)
}
