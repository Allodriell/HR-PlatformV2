import { useLayoutEffect, useRef, useState } from "react";

import { ChevronRightIcon } from "./icons/ChevronRightIcon";
import { UserCircleIcon } from "./icons/UserCircleIcon";

type CandidateCardProps = {
  avatarUrl?: string;
  id: string;
  meta: string;
  name: string;
  onOpen?: (id: string) => void;
  tags: readonly string[];
};

export function CandidateCard({
  avatarUrl,
  id,
  meta,
  name,
  onOpen,
  tags,
}: CandidateCardProps) {
  const tagsRef = useRef<HTMLDivElement>(null);
  const measureRef = useRef<HTMLDivElement>(null);
  const [visibleTagsCount, setVisibleTagsCount] = useState(tags.length);

  useLayoutEffect(() => {
    const tagsNode = tagsRef.current;
    const measureNode = measureRef.current;

    if (!tagsNode || !measureNode) {
      return;
    }

    const calculateVisibleTags = () => {
      const availableWidth = tagsNode.clientWidth;
      const gap = 12;
      const tagWidths = tags.map((_, index) => {
        const node = measureNode.querySelector<HTMLElement>(`[data-tag-index="${index}"]`);
        return node?.offsetWidth ?? 0;
      });
      const moreTagWidth =
        measureNode.querySelector<HTMLElement>("[data-more-tag]")?.offsetWidth ?? 42;

      let usedWidth = 0;
      let nextVisibleCount = 0;

      for (const tagWidth of tagWidths) {
        const nextGap = nextVisibleCount > 0 ? gap : 0;
        const remainingAfterThis = tags.length - nextVisibleCount - 1;
        const reservedMoreWidth = remainingAfterThis > 0 ? gap + moreTagWidth : 0;

        if (usedWidth + nextGap + tagWidth + reservedMoreWidth > availableWidth) {
          break;
        }

        usedWidth += nextGap + tagWidth;
        nextVisibleCount += 1;
      }

      if (nextVisibleCount === 0 && tags.length > 0) {
        nextVisibleCount = 1;
      }

      setVisibleTagsCount(nextVisibleCount);
    };

    calculateVisibleTags();

    const resizeObserver = new ResizeObserver(calculateVisibleTags);
    resizeObserver.observe(tagsNode);

    return () => resizeObserver.disconnect();
  }, [tags]);

  const visibleTags = tags.slice(0, visibleTagsCount);
  const hiddenTagsCount = Math.max(tags.length - visibleTags.length, 0);

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
        <div ref={tagsRef} className="candidate-card__tags-visible">
          {visibleTags.map((tag) => (
            <span className="candidate-tag" key={tag}>
              {tag}
            </span>
          ))}
          {hiddenTagsCount > 0 ? (
            <span className="candidate-tag candidate-tag--more">+{hiddenTagsCount}</span>
          ) : null}
        </div>

        <div aria-hidden="true" className="candidate-card__tags-measure" ref={measureRef}>
          {tags.map((tag, index) => (
            <span className="candidate-tag" data-tag-index={index} key={`${tag}-${index}`}>
              {tag}
            </span>
          ))}
          <span className="candidate-tag candidate-tag--more" data-more-tag>
            +9
          </span>
        </div>
      </div>

      <button
        aria-label={`Open ${name}`}
        className="candidate-card__more"
        onClick={() => onOpen?.(id)}
        type="button"
      >
        <ChevronRightIcon />
      </button>
    </article>
  );
}
