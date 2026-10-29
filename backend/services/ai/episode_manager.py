from datetime import datetime
from uuid import uuid4

from models import Episode


class EpisodeManager:

    def generate_episode_id(self):

        return f"EP_{uuid4().hex[:8].upper()}"


    def get_active_episode(self, db, user_id):

        return db.query(Episode).filter(

            Episode.user_id == user_id,

            Episode.status == "ACTIVE"

        ).first()


    def create_episode(self, db, user_id, title):

        episode = Episode(

            episode_id=self.generate_episode_id(),

            user_id=user_id,

            title=title,

            status="ACTIVE"

        )

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

        episode.closed_at = datetime.utcnow()

        db.commit()

        db.refresh(episode)

        return episode