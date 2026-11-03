from datetime import datetime
from uuid import uuid4

from models import Episode, User


class EpisodeManager:
    def generate_episode_id(self):
        return f"EP_{uuid4().hex[:8].upper()}"

    def get_active_episode(self, db, user_id):
        return (db.query(Episode)
                .filter(Episode.user_id == user_id, Episode.status == "ACTIVE")
                .one_or_none())

    def create_episode(self, db, user_id, title):
        # Lock the parent user row to serialize concurrent episode creation.
        db.query(User).filter(User.id == user_id).with_for_update().one()
        existing = self.get_active_episode(db, user_id)
        if existing is not None:
            return existing
        episode = Episode(episode_id=self.generate_episode_id(), user_id=user_id,
                          title=(title or "Untitled Episode")[:255], status="ACTIVE",
                          active_user_id=user_id)
        db.add(episode)
        db.commit()
        db.refresh(episode)
        return episode

    def update_summary(self, db, episode, summary):
        episode.summary = summary
        db.commit()
        db.refresh(episode)
        return episode

    def close_episode(self, db, episode):
        episode.status = "CLOSED"
        episode.active_user_id = None
        episode.closed_at = datetime.utcnow()
        db.commit()
        db.refresh(episode)
        return episode
