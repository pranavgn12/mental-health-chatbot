from models import EpisodeReflection

from services.ai.memory_consolidator import MemoryConsolidator


class LongTermMemory:

    def __init__(self):

        self.memory_consolidator = MemoryConsolidator()

    def update(

        self,

        db,

        user_id,

        episode

    ):

        ##################################################
        # Load Previous Memory
        ##################################################

        previous_memory = self.memory_consolidator.load(

            db,

            user_id

        )

        ##################################################
        # Load Episode Reflections
        ##################################################

        reflections = db.query(

            EpisodeReflection

        ).filter(

            EpisodeReflection.episode_id == episode.episode_id

        ).order_by(

            EpisodeReflection.created_at

        ).all()

        ##################################################
        # Consolidate
        ##################################################

        memory = self.memory_consolidator.consolidate(

            previous_memory,

            episode.summary,

            reflections

        )

        ##################################################
        # Save
        ##################################################

        self.memory_consolidator.save(

            db,

            user_id,

            memory

        )

        return memory

    def retrieve(

        self,

        db,

        user_id

    ):

        return self.memory_consolidator.load(

            db,

            user_id

        )