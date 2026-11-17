
import json
import hashlib
import logging
import re

from services.llm_provider import generate_response
from services.ai.prompts.profile_generator_prompt import PROMPT

logger = logging.getLogger(__name__)


class Profile:

    MAX_THEMES = 8

    ALLOWED_TONES = {
        "amber",
        "green",
        "blue",
        "purple"
    }

    DEFAULT_TONE = "blue"

    # =====================================================
    # BUILD PROMPT
    # =====================================================

    def build_prompt(self, memory, previous_themes=None):
        previous_themes_json = json.dumps(
            previous_themes or [],
            ensure_ascii=False,
            indent=2
        )

        return (
            PROMPT
            .replace("{memory}", str(memory))
            .replace("{previous_themes}", previous_themes_json)
        )

        return self.PROMPT.format(
            memory=memory.strip(),
            previous_themes=previous_themes_json
        )

    # =====================================================
    # GENERATE PROFILE
    # =====================================================

    def generate(self, memory, previous_themes=None):

        """
        Generate Profile themes from complete long-term memory.

        Args:
            memory:
                Complete consolidated long-term memory.

            previous_themes:
                Previously saved Profile themes.

        Returns:
            List of validated Profile theme dictionaries.
        """

        if not memory or not memory.strip():
            logger.warning(
                "Cannot generate Profile from empty memory."
            )

            return []

        prompt = self.build_prompt(
            memory,
            previous_themes
        )

        logger.info(
            "Generating Profile themes from long-term memory."
        )

        response = generate_response(prompt, max_completion_tokens=4096)

        if not isinstance(response, str) or not response.strip():
            raise ValueError(
                "Profile generator returned an empty response."
            )

        # Parse LLM response.
        parsed = self.parse_response(response)

        # Validate and normalize themes.
        themes = self.validate_themes(parsed)

        if not themes:
            raise ValueError(
                "Profile generator returned no valid themes."
            )

        logger.info(
            "Successfully generated %d Profile themes.",
            len(themes)
        )

        print("\n========== GENERATED PROFILE ==========")

        print(
            json.dumps(
                themes,
                indent=2,
                ensure_ascii=False
            )
        )

        print("=======================================\n")

        return themes

    # =====================================================
    # PARSE JSON RESPONSE
    # =====================================================

    def parse_response(self, response):

        """
        Parse JSON even when the LLM accidentally wraps
        the response inside markdown code fences or text.
        """

        cleaned = response.strip()

        # Remove markdown code fences.
        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE
        )

        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned
        )

        # Find the beginning of the JSON.
        decoder = json.JSONDecoder()

        start_positions = [
            position
            for position in (
                cleaned.find("{"),
                cleaned.find("[")
            )
            if position >= 0
        ]

        if not start_positions:
            raise ValueError(
                "No JSON object or array found in Profile response."
            )

        start = min(start_positions)

        try:

            parsed, _ = decoder.raw_decode(
                cleaned[start:]
            )

            return parsed

        except json.JSONDecodeError as exc:

            logger.exception(
                "Could not parse Profile generator JSON."
            )

            raise ValueError(
                "Invalid JSON returned by Profile generator."
            ) from exc

    # =====================================================
    # VALIDATE AND NORMALIZE THEMES
    # =====================================================

    def validate_themes(self, parsed):

        """
        Validate the theme structure and normalize each theme.

        Generates stable IDs based on category and title.
        """

        if isinstance(parsed, list):
            raw_themes = parsed

        elif isinstance(parsed, dict):
            raw_themes = parsed.get("themes", [])

        else:
            raise ValueError(
                "Invalid Profile JSON structure."
            )

        if not isinstance(raw_themes, list):
            raise ValueError(
                "Profile themes must be a list."
            )

        themes = []

        seen = set()

        for item in raw_themes:

            if not isinstance(item, dict):
                continue

            category = self.clean_text(
                item.get("category"),
                max_length=40
            )

            title = self.clean_text(
                item.get("title"),
                max_length=60
            )

            summary = self.clean_text(
                item.get("summary"),
                max_length=120
            )

            details = self.clean_text(
                item.get("details"),
                max_length=600
            )

            tone = self.clean_text(
                item.get("tone"),
                max_length=20
            ).lower()

            # Required fields.
            if not title or not details:
                continue

            if not category:
                category = "Life theme"

            if not summary:
                summary = self.clean_text(
                    details,
                    max_length=100
                )

            # Only allow known visual tones.
            if tone not in self.ALLOWED_TONES:
                tone = self.DEFAULT_TONE

            # Generate stable ID.
            theme_id = self.generate_id(
                category,
                title
            )

            # Avoid duplicate category/title pairs.
            duplicate_key = (
                category.lower(),
                title.lower()
            )

            if duplicate_key in seen:
                continue

            seen.add(duplicate_key)

            themes.append({
                "id": theme_id,
                "category": category,
                "title": title,
                "summary": summary,
                "details": details,
                "tone": tone
            })

            # Keep Profile manageable.
            if len(themes) >= self.MAX_THEMES:
                break

        return themes

    # =====================================================
    # GENERATE STABLE THEME ID
    # =====================================================

    def generate_id(self, category, title):

        """
        Same category and title produce the same ID.

        This allows React to recognize existing themes
        when new themes are added.
        """

        value = (
            category.strip().lower()
            + "::"
            + title.strip().lower()
        )

        digest = hashlib.sha256(
            value.encode("utf-8")
        ).hexdigest()[:12]

        return f"theme_{digest}"

    # =====================================================
    # CLEAN TEXT
    # =====================================================

    def clean_text(self, value, max_length=500):

        if not isinstance(value, str):
            return ""

        return " ".join(
            value.split()
        )[:max_length].strip()

