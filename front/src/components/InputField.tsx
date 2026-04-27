import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { Button } from "./Button";
import { AgentWaveIcon } from "./icons/AgentWaveIcon";
import { HeadphonesIcon } from "./icons/HeadphonesIcon";
import { MicrophoneIcon } from "./icons/MicrophoneIcon";
import { SearchIcon } from "./icons/SearchIcon";
import { SendArrowIcon } from "./icons/SendArrowIcon";

export type InputMode = "idle" | "typing" | "recording" | "agent";

type InputFieldProps = {
  className?: string;
  compact?: boolean;
  initialValue?: string;
  mode?: InputMode;
  onModeChange?: (mode: InputMode) => void;
  onSubmit?: (value: string) => void;
  onValueChange?: (value: string) => void;
  placeholder?: string;
  value?: string;
};

export function InputField({
  className = "",
  compact = false,
  initialValue = "",
  mode: controlledMode,
  onModeChange,
  onSubmit,
  onValueChange,
  placeholder = "Найти кандидата",
  value: controlledValue,
}: InputFieldProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [uncontrolledValue, setUncontrolledValue] = useState(initialValue);
  const [isFocused, setIsFocused] = useState(false);
  const [isMultiline, setIsMultiline] = useState(false);
  const [uncontrolledMode, setUncontrolledMode] = useState<InputMode>(
    initialValue ? "typing" : "idle",
  );

  const value = controlledValue ?? uncontrolledValue;
  const mode = controlledMode ?? uncontrolledMode;

  const updateMode = (nextMode: InputMode) => {
    if (controlledMode === undefined) {
      setUncontrolledMode(nextMode);
    }

    onModeChange?.(nextMode);
  };

  const updateValue = (nextValue: string) => {
    if (controlledValue === undefined) {
      setUncontrolledValue(nextValue);
    }

    onValueChange?.(nextValue);
  };

  const isRecording = mode === "recording";
  const isAgent = mode === "agent";
  const hasTypedValue = value.trim().length > 0;
  const shouldShowSend = mode === "typing" && hasTypedValue;

  const handleChange = (nextValue: string) => {
    updateValue(nextValue);

    if (isRecording) {
      return;
    }

    if (isAgent) {
      return;
    }

    updateMode(nextValue.trim() ? "typing" : "idle");
  };

  const handleSubmit = () => {
    const trimmedValue = value.trim();

    if (!trimmedValue) {
      return;
    }

    onSubmit?.(trimmedValue);
  };

  const handleRecordClick = () => {
    const nextMode = isRecording ? (value.trim() ? "typing" : "idle") : "recording";
    updateMode(nextMode);
  };

  const handleAgentClick = () => {
    const nextMode = isAgent ? (value.trim() ? "typing" : "idle") : "agent";
    updateMode(nextMode);
  };

  const hideSearchUi =
    mode === "typing"
      ? value.trim().length > 0 || isFocused
      : mode === "idle" && isFocused;

  const leftIcon =
    hideSearchUi ? null : mode === "recording" ? (
      <MicrophoneIcon />
    ) : mode === "agent" ? (
      <HeadphonesIcon />
    ) : (
      <SearchIcon />
    );

  const inputValue = mode === "recording" ? "Идет запись" : mode === "agent" ? "Слушаю" : value;

  const readOnly = isRecording || isAgent;

  useEffect(() => {
    if (controlledValue === undefined || isRecording || isAgent) {
      return;
    }

    updateMode(controlledValue.trim() ? "typing" : "idle");
  }, [controlledValue, isAgent, isRecording]);

  useLayoutEffect(() => {
    const textarea = textareaRef.current;

    if (!textarea) {
      return;
    }

    textarea.style.height = "0px";

    const maxTextareaHeight = 116;
    const nextHeight = Math.min(textarea.scrollHeight, maxTextareaHeight);

    textarea.style.height = `${Math.max(nextHeight, 24)}px`;
    textarea.style.overflowY =
      textarea.scrollHeight > maxTextareaHeight ? "auto" : "hidden";
    setIsMultiline(textarea.scrollHeight > 28);
  }, [inputValue]);

  return (
    <div
      className={[
        "input-field",
        isMultiline ? "input-field--multiline" : "input-field--single-line",
        isAgent ? "input-field--agent" : "",
        compact ? "input-field--compact" : "",
        isRecording ? "input-field--recording" : "",
        className,
      ].join(" ")}
    >
      {!compact ? (
        <label className="input-field__content" htmlFor="candidate-search">
          {leftIcon ? (
            <span
              className={[
                "input-field__leading-icon",
                mode === "typing" ? "" : "input-field__leading-icon--muted",
              ]
                .filter(Boolean)
                .join(" ")}
            >
              {leftIcon}
            </span>
          ) : null}

          <textarea
            className={[
              "input-field__control",
              mode === "typing" ? "input-field__control--value" : "",
            ]
              .filter(Boolean)
              .join(" ")}
            ref={textareaRef}
            onBlur={() => setIsFocused(false)}
            rows={1}
            id="candidate-search"
            onChange={(event) => handleChange(event.target.value)}
            onFocus={() => setIsFocused(true)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                handleSubmit();
              }
            }}
            placeholder={readOnly || isFocused ? "" : placeholder}
            readOnly={readOnly}
            value={inputValue}
          />
        </label>
      ) : null}

      <div className="input-field__actions">
        {!compact ? (
          <Button
            aria-label="Record audio"
            leadingIcon={<MicrophoneIcon />}
            mode="toggle"
            onClick={handleRecordClick}
            pressed={isRecording}
            size="icon"
            visualState={isRecording ? "selected" : "default"}
          />
        ) : null}
        {shouldShowSend ? (
          <Button
            aria-label="Send message"
            leadingIcon={<SendArrowIcon />}
            onClick={handleSubmit}
            size="icon"
          />
        ) : (
          <Button
            aria-label="Agent mode"
            leadingIcon={<AgentWaveIcon />}
            mode="toggle"
            onClick={handleAgentClick}
            pressed={isAgent}
            size="icon"
            visualState={isAgent ? "selected" : "default"}
          />
        )}
      </div>
    </div>
  );
}
