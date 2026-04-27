import {
  useState,
  type ButtonHTMLAttributes,
  type FocusEvent,
  type KeyboardEvent,
  type MouseEvent,
  type PointerEvent,
  type ReactNode,
} from "react";

type ButtonMode = "toggle" | "press";
type ButtonSize = "md" | "icon";
type ButtonVisualState = "default" | "hover" | "selected" | "active";

type ButtonProps = Omit<ButtonHTMLAttributes<HTMLButtonElement>, "onChange"> & {
  defaultActive?: boolean;
  leadingIcon?: ReactNode;
  mode?: ButtonMode;
  onActiveChange?: (active: boolean) => void;
  pressed?: boolean;
  size?: ButtonSize;
  trailingIcon?: ReactNode;
  visualState?: ButtonVisualState;
};

export function Button({
  children,
  className = "",
  defaultActive = false,
  leadingIcon,
  mode = "press",
  onActiveChange,
  pressed,
  size = "md",
  trailingIcon,
  type = "button",
  visualState,
  ...props
}: ButtonProps) {
  const [isActive, setIsActive] = useState(defaultActive);
  const resolvedActive = pressed ?? isActive;

  const setActive = (nextActive: boolean) => {
    setIsActive(nextActive);
    onActiveChange?.(nextActive);
  };

  const handleClick = (event: MouseEvent<HTMLButtonElement>) => {
    props.onClick?.(event);

    if (mode === "toggle") {
      setActive(!resolvedActive);
    }
  };

  const handlePointerDown = (event: PointerEvent<HTMLButtonElement>) => {
    props.onPointerDown?.(event);
    if (mode === "press") {
      setActive(true);
    }
  };

  const handlePointerUp = (event: PointerEvent<HTMLButtonElement>) => {
    props.onPointerUp?.(event);
    if (mode === "press") {
      setActive(false);
    }
  };

  const handlePointerLeave = (event: PointerEvent<HTMLButtonElement>) => {
    props.onPointerLeave?.(event);
    if (mode === "press") {
      setActive(false);
    }
  };

  const handleBlur = (event: FocusEvent<HTMLButtonElement>) => {
    props.onBlur?.(event);
    if (mode === "press") {
      setActive(false);
    }
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLButtonElement>) => {
    props.onKeyDown?.(event);
    if (mode === "press" && (event.key === " " || event.key === "Enter")) {
      setActive(true);
    }
  };

  const handleKeyUp = (event: KeyboardEvent<HTMLButtonElement>) => {
    props.onKeyUp?.(event);
    if (mode === "press" && (event.key === " " || event.key === "Enter")) {
      setActive(false);
    }
  };

  const buttonClassName = [
    "ui-button",
    `ui-button--${size}`,
    `ui-button--state-${visualState ?? (resolvedActive ? "selected" : "default")}`,
    size !== "icon" ? "ui-button--with-label" : "",
    className,
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <button
      {...props}
      aria-pressed={mode === "toggle" ? resolvedActive : undefined}
      className={buttonClassName}
      onBlur={handleBlur}
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      onKeyUp={handleKeyUp}
      onPointerDown={handlePointerDown}
      onPointerLeave={handlePointerLeave}
      onPointerUp={handlePointerUp}
      type={type}
    >
      {leadingIcon ? <span className="ui-button__icon">{leadingIcon}</span> : null}
      {size !== "icon" ? <span className="ui-button__label">{children}</span> : null}
      {trailingIcon ? <span className="ui-button__icon">{trailingIcon}</span> : null}
    </button>
  );
}
