import { useState } from "react";
import { ApiError, candidatesApi } from "./api";
import type { Candidate } from "./api";
import { CandidateCard } from "./components/CandidateCard";
import { InputField } from "./components/InputField";
import { CopyIcon } from "./components/icons/CopyIcon";
import { EditIcon } from "./components/icons/EditIcon";
import { TrashIcon } from "./components/icons/TrashIcon";

type CandidateCardViewModel = {
  id: string;
  meta: string;
  name: string;
  tags: readonly string[];
};

function toCandidateCard(candidate: Candidate): CandidateCardViewModel {
  const metaParts = [candidate.headline, candidate.location].filter(Boolean);
  const scoreLabel =
    typeof candidate.score === "number" ? `score ${candidate.score.toFixed(2)}` : undefined;

  return {
    id: candidate.id,
    name: candidate.fullName,
    meta: [metaParts.join(" • "), scoreLabel].filter(Boolean).join(" • ") || "Кандидат из базы",
    tags: candidate.headline ? [candidate.headline] : [],
  };
}

export default function App() {
  const [query, setQuery] = useState("");
  const [submittedPrompt, setSubmittedPrompt] = useState("");
  const [isEditingPrompt, setIsEditingPrompt] = useState(false);
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  const hasResults = submittedPrompt.trim().length > 0;

  const handleSubmit = async (value: string) => {
    setSubmittedPrompt(value);
    setQuery("");
    setIsEditingPrompt(false);
    setIsLoading(true);
    setErrorMessage("");

    try {
      const response = await candidatesApi.search({
        limit: 10,
        query: value,
      });
      setCandidates(response.items);
    } catch (error) {
      const message =
        error instanceof ApiError ? error.message : "Не удалось получить кандидатов.";
      setCandidates([]);
      setErrorMessage(message);
    } finally {
      setIsLoading(false);
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
    setIsEditingPrompt(false);
    setCandidates([]);
    setErrorMessage("");
  };

  return (
    <main className={["search-screen", hasResults ? "search-screen--results" : ""].join(" ")}>
      <section className={["search-layout", hasResults ? "search-layout--results" : ""].join(" ")}>
        <div className="search-left">
          {!hasResults ? (
            <header className="search-hero">
              <h1 className="search-hero__title">Добрый день, кого ищем сегодня?</h1>
            </header>
          ) : null}

          {hasResults ? (
            <section className="prompt-panel">
              <div className="prompt-panel__card">
                {isEditingPrompt ? (
                  <textarea
                    className="prompt-panel__editor"
                    onChange={(event) => setSubmittedPrompt(event.target.value)}
                    rows={7}
                    value={submittedPrompt}
                  />
                ) : (
                  <p className="prompt-panel__text">{submittedPrompt}</p>
                )}
              </div>

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
            </section>
          ) : null}

          <div className="search-input-wrap">
            <InputField
              onSubmit={handleSubmit}
              onValueChange={setQuery}
              placeholder="Найти кандидата"
              value={query}
            />
          </div>
        </div>

        {hasResults ? (
          <aside className="candidate-column">
            {isLoading ? <p>Загрузка кандидатов...</p> : null}
            {!isLoading && errorMessage ? <p>{errorMessage}</p> : null}
            {!isLoading && !errorMessage && candidates.length === 0 ? (
              <p>Кандидаты не найдены.</p>
            ) : null}
            {!isLoading && !errorMessage
              ? candidates.map((candidate) => (
                  <CandidateCard key={candidate.id} {...toCandidateCard(candidate)} />
                ))
              : null}
          </aside>
        ) : null}
      </section>
    </main>
  );
}
