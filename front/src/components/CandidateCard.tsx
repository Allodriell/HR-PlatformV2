import { ChevronRightIcon } from "./icons/ChevronRightIcon";
import { UserCircleIcon } from "./icons/UserCircleIcon";

type CandidateCardProps = {
  avatarUrl?: string;
  id: string;
  meta: string;
  name: string;
  tags: readonly string[];
};

export function CandidateCard({
  avatarUrl,
  id,
  meta,
  name,
  tags,
}: CandidateCardProps) {
  return (
    <article className="candidate-card" data-candidate-id={id}>
      <div aria-hidden="true" className="candidate-card__avatar">
        {avatarUrl ? (
          <img
            alt=""
            className="candidate-card__avatar-image"
            loading="lazy"
            src={avatarUrl}
          />
        ) : (
          <UserCircleIcon />
        )}
      </div>

      <div className="candidate-card__content">
        <h2 className="candidate-card__name">{name}</h2>
        <p className="candidate-card__meta">{meta}</p>
      </div>

      <div className="candidate-card__tags">
        {tags.map((tag) => (
          <span className="candidate-tag" key={tag}>
            {tag}
          </span>
        ))}
      </div>

      <button
        aria-label={`Open ${name}`}
        className="candidate-card__more"
        type="button"
      >
        <ChevronRightIcon />
      </button>
    </article>
  );
}
