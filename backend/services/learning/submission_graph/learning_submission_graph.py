from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from songguo.backend.services.learning.llm_session_runner import LLMSessionRunner
from songguo.backend.services.learning.math_structuring import (
    MathProblemStructuringGateway,
)
from songguo.backend.services.learning.store import InMemoryLearningStore
from songguo.backend.services.learning.submission_graph.nodes import (
    LearningSubmissionGraphNodes,
    input_normalize_node,
)
from songguo.backend.services.learning.submission_graph.state import (
    LearningSubmissionGraphResult,
    LearningSubmissionGraphState,
)
from songguo.backend.services.learning.submission_models import (
    LearningItemStatus,
    LearningSubmissionStatus,
)
from songguo.backend.services.learning.tutor_graph.math_mistake_graph import (
    MathMistakeTutorGraph,
)


class LearningSubmissionGraph:
    def __init__(
        self,
        *,
        store: InMemoryLearningStore,
        session_runner: LLMSessionRunner | None = None,
        math_gateway: MathProblemStructuringGateway | None = None,
        tutor_graph: MathMistakeTutorGraph | None = None,
    ) -> None:
        self.store = store
        self.math_gateway = math_gateway or MathProblemStructuringGateway()
        self.tutor_graph = tutor_graph or MathMistakeTutorGraph(
            store=store,
            session_runner=session_runner,
            math_gateway=self.math_gateway,
        )
        self.nodes = LearningSubmissionGraphNodes(
            store=self.store,
            math_gateway=self.math_gateway,
            tutor_graph=self.tutor_graph,
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(LearningSubmissionGraphState)
        graph.add_node("input_normalize", input_normalize_node)
        graph.add_node("intake_parse", self.nodes.intake_parse_node)
        graph.add_node("structure_and_judge_items", self.nodes.structure_and_judge_items_node)
        graph.add_node("start_or_resume_active_wrong_item", self.nodes.start_or_resume_active_wrong_item_node)
        graph.add_node("build_submission_summary", self.nodes.build_submission_summary_node)
        graph.add_edge(START, "input_normalize")
        graph.add_edge("input_normalize", "intake_parse")
        graph.add_edge("intake_parse", "structure_and_judge_items")
        graph.add_edge("structure_and_judge_items", "start_or_resume_active_wrong_item")
        graph.add_edge("start_or_resume_active_wrong_item", "build_submission_summary")
        graph.add_edge("build_submission_summary", END)
        return graph.compile()

    def start(
        self,
        *,
        child_id: str,
        subject: str,
        grade: int,
        source_type: str,
        raw_text: str,
    ) -> LearningSubmissionGraphResult:
        return self.create_draft(
            child_id=child_id,
            subject=subject,
            grade=grade,
            source_type=source_type,
            raw_text=raw_text,
        )

    def create_draft(
        self,
        *,
        child_id: str,
        subject: str,
        grade: int,
        source_type: str,
        raw_text: str,
    ) -> LearningSubmissionGraphResult:
        state = LearningSubmissionGraphState(
            child_id=child_id,
            subject=subject or "math",
            grade=grade,
            source_type=source_type,
            raw_text=raw_text,
        )
        state = input_normalize_node(state)
        state = self.nodes.intake_parse_node(state)
        state = self.nodes.build_submission_summary_node(state)
        return self._result_from_state(state)

    def confirm(self, submission_id: str) -> LearningSubmissionGraphResult:
        submission = self.store.require_submission(submission_id)
        state = LearningSubmissionGraphState(
            submission_id=submission_id,
            child_id=submission.child_id,
            subject=submission.subject,
            grade=submission.grade,
            source_type=str(submission.source_type),
            raw_text=submission.raw_text,
            status=submission.status,
        )
        if submission.status == LearningSubmissionStatus.INTAKE_PENDING:
            state = self.nodes.structure_and_judge_items_node(state)
        state = self.nodes.start_or_resume_active_wrong_item_node(state)
        state = self.nodes.build_submission_summary_node(state)
        return self._result_from_state(state)

    def complete_active_tutor_item(
        self,
        *,
        submission_id: str,
        tutor_session_id: str | None = None,
    ) -> LearningSubmissionGraphResult:
        queue_item = self.store.get_active_tutor_item(submission_id)
        if queue_item is None:
            submission = self.store.complete_submission_if_queue_done(submission_id)
            return self._result_from_submission(submission)
        if tutor_session_id and queue_item.tutor_session_id != tutor_session_id:
            raise ValueError("tutor_session_id does not match active queue item")
        self.store.mark_tutor_item_completed(queue_item.queue_item_id, tutor_session_id=tutor_session_id)
        self.store.update_submission_item(queue_item.item_id, status=LearningItemStatus.COMPLETED)
        state = LearningSubmissionGraphState(
            submission_id=submission_id,
            child_id=queue_item.child_id,
            subject=self.store.require_submission(submission_id).subject,
            grade=self.store.require_submission(submission_id).grade,
        )
        state = self.nodes.start_or_resume_active_wrong_item_node(state)
        state = self.nodes.build_submission_summary_node(state)
        return self._result_from_state(state)

    def start_next_tutor_item(self, submission_id: str) -> LearningSubmissionGraphResult:
        submission = self.store.require_submission(submission_id)
        state = LearningSubmissionGraphState(
            submission_id=submission_id,
            child_id=submission.child_id,
            subject=submission.subject,
            grade=submission.grade,
            status=submission.status,
        )
        state = self.nodes.start_or_resume_active_wrong_item_node(state)
        state = self.nodes.build_submission_summary_node(state)
        return self._result_from_state(state)

    def _result_from_state(self, state: LearningSubmissionGraphState) -> LearningSubmissionGraphResult:
        if not state.submission_id:
            raise RuntimeError("LearningSubmissionGraph did not create a submission")
        return LearningSubmissionGraphResult(
            submission_id=state.submission_id,
            status=state.status,
            item_count=state.item_count,
            correct_count=state.correct_count,
            wrong_count=state.wrong_count,
            active_queue_item_id=state.active_queue_item_id,
            active_tutor_session_id=state.active_tutor_session_id,
            summary=state.summary,
        )

    def _result_from_submission(self, submission) -> LearningSubmissionGraphResult:
        return LearningSubmissionGraphResult(
            submission_id=submission.submission_id,
            status=submission.status,
            item_count=submission.item_count,
            correct_count=submission.correct_count,
            wrong_count=submission.wrong_count,
            active_queue_item_id=submission.active_queue_item_id,
        )
