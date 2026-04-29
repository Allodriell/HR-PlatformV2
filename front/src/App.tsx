import { type FormEvent, useState } from "react";
import { ApiError, assistantApi, candidatesApi } from "./api";
import type { Candidate, CandidateDetailResponse } from "./api";
import { CandidateCard } from "./components/CandidateCard";
import { InputField } from "./components/InputField";
import { ArrowLeftIcon } from "./components/icons/ArrowLeftIcon";
import { CopyIcon } from "./components/icons/CopyIcon";
import { EditIcon } from "./components/icons/EditIcon";
import { TrashIcon } from "./components/icons/TrashIcon";
import { UserCircleIcon } from "./components/icons/UserCircleIcon";

type CandidateCardViewModel = {
  id: string;
  meta: string;
  name: string;
  tags: readonly string[];
};

type LoadingStage = "idle" | "thinking" | "searching";

function toCandidateCard(candidate: Candidate): CandidateCardViewModel {
  const metaParts = [candidate.headline, candidate.location].filter(Boolean);
  const scoreLabel =
    typeof candidate.score === "number" ? `${candidate.score.toFixed(3)}p` : undefined;

  return {
    id: candidate.id,
    name: candidate.fullName,
    meta: [metaParts.join(" • "), scoreLabel].filter(Boolean).join(" • ") || "Кандидат из базы",
    tags: candidate.tags?.length ? candidate.tags : candidate.headline ? [candidate.headline] : [],
  };
}

function HighlightedResumeText({
  highlight,
  text,
}: {
  highlight: string;
  text: string;
}) {
  const normalizedHighlight = highlight.trim();

  if (!normalizedHighlight) {
    return <>{text}</>;
  }

  const matchIndex = text.toLowerCase().indexOf(normalizedHighlight.toLowerCase());

  if (matchIndex !== -1) {
    const before = text.slice(0, matchIndex);
    const match = text.slice(matchIndex, matchIndex + normalizedHighlight.length);
    const after = text.slice(matchIndex + normalizedHighlight.length);

    return (
      <>
        {before}
        <mark className="resume-highlight">{match}</mark>
        {after}
      </>
    );
  }

  const textMap = normalizeTextForSearch(text);
  const highlightMap = normalizeTextForSearch(normalizedHighlight);
  const fuzzyMatchIndex = textMap.value.indexOf(highlightMap.value);

  if (fuzzyMatchIndex === -1) {
    return <>{text}</>;
  }

  const matchStart = textMap.indexes[fuzzyMatchIndex] ?? 0;
  const lastNormalizedIndex = fuzzyMatchIndex + highlightMap.value.length - 1;
  const matchEnd = (textMap.indexes[lastNormalizedIndex] ?? matchStart) + 1;
  const before = text.slice(0, matchStart);
  const match = text.slice(matchStart, matchEnd);
  const after = text.slice(matchEnd);

  return (
    <>
      {before}
      <mark className="resume-highlight">{match}</mark>
      {after}
    </>
  );
}

function normalizeTextForSearch(value: string) {
  let normalized = "";
  const indexes: number[] = [];
  let previousWasSpace = false;

  Array.from(value).forEach((char, index) => {
    if (/\s/.test(char)) {
      if (!previousWasSpace && normalized.length > 0) {
        normalized += " ";
        indexes.push(index);
      }
      previousWasSpace = true;
      return;
    }

    normalized += char.toLowerCase();
    indexes.push(index);
    previousWasSpace = false;
  });

  return {
    indexes,
    value: normalized.trim(),
  };
}

function LoadingState({ label }: { label: string }) {
  return (
    <div className="loading-state" role="status">
      <p>{label}</p>
      <div className="loading-state__bar" aria-hidden="true">
        <span />
      </div>
    </div>
  );
}

function QueryPanel({
  isEditing,
  onChange,
  text,
}: {
  isEditing: boolean;
  onChange: (value: string) => void;
  text: string;
}) {
  return (
    <div className="query-panel">
      <div className="query-panel__label">Запрос</div>
      {isEditing ? (
        <textarea
          className="query-panel__editor"
          onChange={(event) => onChange(event.target.value)}
          rows={7}
          value={text}
        />
      ) : (
        <p className="query-panel__text">{text}</p>
      )}
    </div>
  );
}

export default function App() {
  const [mode, setMode] = useState<"search" | "create">("search");
  const [query, setQuery] = useState("");
  const [draftPrompt, setDraftPrompt] = useState("");
  const [submittedPrompt, setSubmittedPrompt] = useState("");
  const [agentPrompt, setAgentPrompt] = useState("Добрый день, кого ищем сегодня?");
  const [isEditingPrompt, setIsEditingPrompt] = useState(false);
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [selectedCandidate, setSelectedCandidate] = useState<CandidateDetailResponse | null>(null);
  const [candidateQaAnswer, setCandidateQaAnswer] = useState("");
  const [candidateQaHistory, setCandidateQaHistory] = useState<
    Array<{ content: string; role: "user" | "assistant" }>
  >([]);
  const [highlightedResumeQuote, setHighlightedResumeQuote] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [loadingStage, setLoadingStage] = useState<LoadingStage>("idle");
  const [isCandidateLoading, setIsCandidateLoading] = useState(false);
  const [isCandidateQaLoading, setIsCandidateQaLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [createForm, setCreateForm] = useState({
    email: "",
    fullName: "",
    phone: "",
    rawResumeText: "",
    role: "",
  });
  const [isCreating, setIsCreating] = useState(false);
  const [createMessage, setCreateMessage] = useState("");

  const hasResults = submittedPrompt.trim().length > 0;
  const displayedPrompt = submittedPrompt || draftPrompt;
  const hasDraftPrompt = draftPrompt.trim().length > 0;
  const hasDisplayedPrompt = displayedPrompt.trim().length > 0;
  const canGoBack =
    mode === "create" ||
    selectedCandidate !== null ||
    candidates.length > 0 ||
    hasDisplayedPrompt ||
    Boolean(candidateQaAnswer);

  const switchMode = (nextMode: "search" | "create") => {
    setMode(nextMode);
    setErrorMessage("");
    setCreateMessage("");
  };

  const handleBack = () => {
    if (selectedCandidate) {
      setSelectedCandidate(null);
      setCandidateQaAnswer("");
      setCandidateQaHistory([]);
      setHighlightedResumeQuote("");
      setIsCandidateQaLoading(false);
      setErrorMessage("");
      return;
    }

    if (mode === "create") {
      switchMode("search");
      return;
    }

    handleDelete();
  };

  const handleSubmit = async (value: string) => {
    if (selectedCandidate) {
      const question = value.trim();

      if (!question) {
        return;
      }

      setQuery("");
      setIsCandidateQaLoading(true);
      setErrorMessage("");

      try {
        const response = await candidatesApi.askQuestion(
          selectedCandidate.candidate.candidate_id,
          {
            history: candidateQaHistory,
            question,
          },
        );

        setCandidateQaAnswer(response.answer);
        setHighlightedResumeQuote(response.evidence_quote ?? "");
        setCandidateQaHistory([
          ...candidateQaHistory,
          { role: "user", content: question },
          { role: "assistant", content: response.answer },
        ]);
      } catch (error) {
        const message =
          error instanceof ApiError ? error.message : "Не удалось получить ответ по резюме.";
        setErrorMessage(message);
      } finally {
        setIsCandidateQaLoading(false);
      }
      return;
    }

    const enteredText = value.trim() || query.trim();

    if (!enteredText && hasDisplayedPrompt) {
      await handleForceSearch();
      return;
    }

    if (!enteredText) {
      return;
    }

    const message = enteredText;
    const currentPrompt = hasResults ? submittedPrompt : draftPrompt;

    setQuery("");
    setIsEditingPrompt(false);
    setIsLoading(true);
    setLoadingStage("thinking");
    setErrorMessage("");

    try {
      const response = await assistantApi.sendPrompt({
        current_prompt: currentPrompt || undefined,
        decide_only: true,
        message,
        mode: "agent",
      });

      const nextPrompt =
        response.normalized_query ||
        response.search?.normalized_query ||
        currentPrompt ||
        message;

      if (response.action === "needs_clarification") {
        setDraftPrompt(nextPrompt);
        setAgentPrompt(response.answer);
        setSubmittedPrompt("");
        setCandidates([]);
        return;
      }

      const normalizedQuery = response.search?.normalized_query || response.normalized_query || nextPrompt;
      setSubmittedPrompt(normalizedQuery);
      setDraftPrompt("");
      setAgentPrompt("Добрый день, кого ищем сегодня?");
      setLoadingStage("searching");

      const searchResponse = await candidatesApi.search({
        limit: 10,
        query: normalizedQuery,
      });
      const finalQuery = searchResponse.meta?.normalized_query?.trim() || normalizedQuery;

      setSubmittedPrompt(finalQuery);
      setCandidates(searchResponse.items);
      setSelectedCandidate(null);
      setCandidateQaAnswer("");
      setCandidateQaHistory([]);
      setHighlightedResumeQuote("");
    } catch (error) {
      const message =
        error instanceof ApiError ? error.message : "Не удалось получить кандидатов.";
      setCandidates([]);
      setErrorMessage(message);
    } finally {
      setIsLoading(false);
      setLoadingStage("idle");
    }
  };

  const handleForceSearch = async () => {
    const forcedQuery = hasResults ? submittedPrompt : draftPrompt || query;
    const normalizedForcedQuery = forcedQuery.trim();

    if (!normalizedForcedQuery) {
      return;
    }

    setSubmittedPrompt(normalizedForcedQuery);
    setQuery("");
    setIsLoading(true);
    setLoadingStage("searching");
    setErrorMessage("");

    try {
      const response = await candidatesApi.search({
        limit: 10,
        query: normalizedForcedQuery,
      });
      const finalQuery = response.meta?.normalized_query?.trim() || normalizedForcedQuery;

      setSubmittedPrompt(finalQuery);
      setCandidates(response.items);
      setSelectedCandidate(null);
      setCandidateQaAnswer("");
      setCandidateQaHistory([]);
      setHighlightedResumeQuote("");
      setDraftPrompt("");
      setAgentPrompt("Добрый день, кого ищем сегодня?");
    } catch (error) {
      const message =
        error instanceof ApiError ? error.message : "Не удалось получить кандидатов.";
      setCandidates([]);
      setErrorMessage(message);
    } finally {
      setIsLoading(false);
      setLoadingStage("idle");
    }
  };

  const handleCopy = async () => {
    if (!submittedPrompt.trim()) {
      return;
    }

    try {
      await navigator.clipboard.writeText(submittedPrompt);
    } catch {
      // noop for local flow
    }
  };

  const handleDelete = () => {
    setSubmittedPrompt("");
    setDraftPrompt("");
    setAgentPrompt("Добрый день, кого ищем сегодня?");
    setIsEditingPrompt(false);
    setLoadingStage("idle");
    setCandidates([]);
    setSelectedCandidate(null);
    setCandidateQaAnswer("");
    setCandidateQaHistory([]);
    setHighlightedResumeQuote("");
    setIsCandidateQaLoading(false);
    setErrorMessage("");
  };

  const handleOpenCandidate = async (candidateId: string) => {
    setIsCandidateLoading(true);
    setErrorMessage("");

    try {
      const response = await candidatesApi.get(candidateId);
      setSelectedCandidate(response);
      setCandidateQaAnswer("");
      setCandidateQaHistory([]);
      setHighlightedResumeQuote("");
      setIsCandidateQaLoading(false);
    } catch (error) {
      const message =
        error instanceof ApiError ? error.message : "Не удалось открыть резюме.";
      setErrorMessage(message);
    } finally {
      setIsCandidateLoading(false);
    }
  };

  const handleCreateCandidate = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    const fullName = createForm.fullName.trim();
    const rawResumeText = createForm.rawResumeText.trim();

    if (!fullName || !rawResumeText) {
      setCreateMessage("");
      setErrorMessage("Нужны ФИО и текст резюме.");
      return;
    }

    setIsCreating(true);
    setErrorMessage("");
    setCreateMessage("");

    try {
      const response = await candidatesApi.create({
        email: createForm.email.trim(),
        full_name: fullName,
        phone: createForm.phone.trim(),
        raw_resume_text: rawResumeText,
        role: createForm.role.trim(),
      });

      const tags = response.candidate.tags?.length
        ? ` Теги: ${response.candidate.tags.join(", ")}.`
        : "";
      setCreateMessage(
        `Кандидат добавлен. Чанков: ${response.candidate.chunks_count}.${tags}`,
      );
      setCreateForm({
        email: "",
        fullName: "",
        phone: "",
        rawResumeText: "",
        role: "",
      });
    } catch (error) {
      const message =
        error instanceof ApiError ? error.message : "Не удалось добавить резюме.";
      setErrorMessage(message);
    } finally {
      setIsCreating(false);
    }
  };

  return (
    <main
      className={[
        "search-screen",
        hasResults && mode === "search" ? "search-screen--results" : "",
        mode === "create" ? "search-screen--create" : "",
      ].join(" ")}
    >
      <header className="app-header">
        {canGoBack ? (
          <button
            aria-label="Назад"
            className="back-button"
            onClick={handleBack}
            type="button"
          >
            <ArrowLeftIcon />
          </button>
        ) : (
          <span aria-hidden="true" className="app-header__spacer" />
        )}

        <nav aria-label="Режим работы" className="mode-switch">
          <button
            className={["mode-switch__button", mode === "search" ? "mode-switch__button--active" : ""].join(" ")}
            onClick={() => switchMode("search")}
            type="button"
          >
            Поиск
          </button>
          <button
            className={["mode-switch__button", mode === "create" ? "mode-switch__button--active" : ""].join(" ")}
            onClick={() => switchMode("create")}
            type="button"
          >
            Резюме
          </button>
        </nav>

        <span aria-hidden="true" className="app-header__spacer" />
      </header>

      {mode === "create" ? (
        <section className="resume-create">
          {isCreating ? (
            <LoadingState label="Загрузка резюме..." />
          ) : (
          <form className="resume-form" onSubmit={handleCreateCandidate}>
            <input
              className="resume-form__control"
              onChange={(event) =>
                setCreateForm((current) => ({ ...current, fullName: event.target.value }))
              }
              placeholder="ФИО"
              value={createForm.fullName}
            />
            <input
              className="resume-form__control"
              onChange={(event) =>
                setCreateForm((current) => ({ ...current, role: event.target.value }))
              }
              placeholder="Роль, если уже известна"
              value={createForm.role}
            />
            <div className="resume-form__row">
              <input
                className="resume-form__control"
                onChange={(event) =>
                  setCreateForm((current) => ({ ...current, email: event.target.value }))
                }
                placeholder="Email"
                value={createForm.email}
              />
              <input
                className="resume-form__control"
                onChange={(event) =>
                  setCreateForm((current) => ({ ...current, phone: event.target.value }))
                }
                placeholder="Телефон"
                value={createForm.phone}
              />
            </div>
            <textarea
              className="resume-form__textarea"
              onChange={(event) =>
                setCreateForm((current) => ({ ...current, rawResumeText: event.target.value }))
              }
              placeholder="Вставь текст резюме"
              value={createForm.rawResumeText}
            />
            <button className="resume-form__submit" disabled={isCreating} type="submit">
              {isCreating ? "Индексируем..." : "Добавить резюме"}
            </button>
          </form>
          )}

          {errorMessage ? <p className="form-status form-status--error">{errorMessage}</p> : null}
          {createMessage ? <p className="form-status">{createMessage}</p> : null}
        </section>
      ) : (
      <section className={["search-layout", hasResults ? "search-layout--results" : ""].join(" ")}>
        <div className="search-left">
          {!hasResults ? (
            <header className="search-hero">
              {hasDraftPrompt ? (
                <QueryPanel
                  isEditing={isEditingPrompt}
                  onChange={setDraftPrompt}
                  text={draftPrompt}
                />
              ) : null}
              {isLoading && loadingStage === "searching" ? (
                <LoadingState label="Загрузка кандидатов..." />
              ) : (
                <h1 className={["search-hero__title", isLoading ? "thinking-text" : ""].join(" ")}>
                  {isLoading ? "Думаю..." : agentPrompt}
                </h1>
              )}
            </header>
          ) : null}

          {hasDisplayedPrompt && hasResults ? (
            <section className="prompt-panel">
              <QueryPanel
                isEditing={isEditingPrompt}
                onChange={setSubmittedPrompt}
                text={displayedPrompt}
              />

              <div className="prompt-panel__actions">
                <button
                  aria-label="Edit prompt"
                  className="prompt-action"
                  onClick={() => setIsEditingPrompt((current) => !current)}
                  type="button"
                >
                  <EditIcon />
                </button>
                <button
                  aria-label="Copy prompt"
                  className="prompt-action"
                  onClick={handleCopy}
                  type="button"
                >
                  <CopyIcon />
                </button>
                <button
                  aria-label="Delete prompt"
                  className="prompt-action prompt-action--danger"
                  onClick={handleDelete}
                  type="button"
                >
                  <TrashIcon />
                </button>
              </div>
              {selectedCandidate && candidateQaAnswer ? (
                <div className="candidate-answer-bubble">
                  {candidateQaAnswer}
                </div>
              ) : null}
              {selectedCandidate && isCandidateQaLoading ? (
                <div className="candidate-answer-bubble candidate-answer-bubble--loading">
                  Ищу ответ в резюме...
                </div>
              ) : null}
            </section>
          ) : null}

          <div className="search-input-wrap">
            <div className={["search-composer", hasDraftPrompt ? "search-composer--active" : ""].join(" ")}>
              <InputField
                onSubmit={handleSubmit}
                onValueChange={setQuery}
                placeholder={hasDraftPrompt ? "Ответить или начать поиск" : "Найти кандидата"}
                submitWhenEmpty={hasDisplayedPrompt}
                value={query}
              />
            </div>
          </div>
        </div>

        {hasResults ? (
          <aside className={selectedCandidate ? "resume-detail-column" : "candidate-column"}>
            {isLoading && !selectedCandidate ? <LoadingState label="Загрузка кандидатов..." /> : null}
            {isCandidateLoading ? <LoadingState label="Загрузка резюме..." /> : null}
            {!isLoading && !isCandidateLoading && errorMessage ? <p>{errorMessage}</p> : null}
            {!isCandidateLoading && !errorMessage && selectedCandidate ? (
              <section className="resume-detail">
                <header className="resume-detail__header">
                  <div aria-hidden="true" className="resume-detail__avatar">
                    <UserCircleIcon />
                  </div>

                  <div className="resume-detail__identity">
                    <h2>{selectedCandidate.candidate.full_name}</h2>
                    <p>
                      {[selectedCandidate.candidate.role, "1.425"]
                        .filter(Boolean)
                        .join(" • ")}
                    </p>
                  </div>

                  <div className="resume-detail__contacts">
                    {selectedCandidate.candidate.email ? (
                      <p>{selectedCandidate.candidate.email}</p>
                    ) : null}
                    {selectedCandidate.candidate.phone ? (
                      <p>{selectedCandidate.candidate.phone}</p>
                    ) : null}
                  </div>
                </header>

                <div className="resume-detail__tags">
                  {(selectedCandidate.candidate.tags ?? []).map((tag) => (
                    <span className="resume-detail__tag" key={tag}>
                      {tag}
                    </span>
                  ))}
                </div>

                <article className="resume-detail__text">
                  <HighlightedResumeText
                    highlight={highlightedResumeQuote}
                    text={selectedCandidate.resume_text}
                  />
                </article>
              </section>
            ) : null}
            {!isLoading && !isCandidateLoading && !errorMessage && candidates.length === 0 && hasDraftPrompt ? (
              <div className="candidate-column__clarification">
                <p>{agentPrompt}</p>
              </div>
            ) : null}
            {!isLoading && !isCandidateLoading && !errorMessage && candidates.length === 0 ? (
              !hasDraftPrompt ? <p>Кандидаты не найдены.</p> : null
            ) : null}
            {!isLoading && !isCandidateLoading && !errorMessage && !selectedCandidate
              ? candidates.map((candidate) => (
                  <CandidateCard
                    key={candidate.id}
                    {...toCandidateCard(candidate)}
                    onOpen={handleOpenCandidate}
                  />
                ))
              : null}
          </aside>
        ) : null}
      </section>
      )}
    </main>
  );
}
