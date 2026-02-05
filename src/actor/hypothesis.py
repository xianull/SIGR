"""
Hypothesis Ledger for SIGR Actor

科学假设账本 (Scientific Hypothesis Ledger)
==========================================

核心理念：Actor 是计算生物学家，通过假设驱动的实验进行策略优化。
不同于参数优化器，生物学家会：
1. 提出可证伪的假设 (Propose falsifiable hypotheses)
2. 设计实验验证假设 (Design experiments to test hypotheses)
3. 根据结果更新认知 (Update understanding based on results)

HypothesisLedger 追踪所有假设的生命周期：
- PROPOSED: 假设已提出，等待验证
- VALIDATED: 假设被实验证实
- INVALIDATED: 假设被实验证伪

这种设计让 Actor 避免重复失败的假设，并积累成功的知识。
"""

import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class HypothesisStatus(str, Enum):
    """假设状态枚举"""
    PROPOSED = "PROPOSED"           # 已提出，待验证
    VALIDATED = "VALIDATED"         # 已验证（实验成功）
    INVALIDATED = "INVALIDATED"     # 已证伪（实验失败）


@dataclass
class Hypothesis:
    """
    科学假设 (Scientific Hypothesis)

    一个完整的科学假设包含：
    - 陈述 (statement): 假设的核心主张
    - 生物学依据 (biological_basis): 为什么从生物学角度这样认为
    - 预期结果 (expected_outcome): 如果假设正确，预期会发生什么
    - 证伪条件 (falsification_criteria): 什么情况下认为假设错误

    Example:
        Hypothesis(
            statement="Reducing neighborhood to marker genes will improve cell classification",
            biological_basis="Cell identity is defined by few specific markers, not the entire interactome",
            expected_outcome="AUC should increase by at least 5%",
            falsification_criteria="If AUC decreases or stays flat, hypothesis is invalid"
        )
    """
    id: str                                                # 唯一标识符 (e.g., "H001")
    statement: str                                         # 假设陈述
    biological_basis: str                                  # 生物学依据
    expected_outcome: str                                  # 预期结果
    falsification_criteria: str                            # 证伪条件
    status: HypothesisStatus = HypothesisStatus.PROPOSED   # 当前状态
    iteration_proposed: int = 0                            # 提出时的迭代
    iteration_resolved: Optional[int] = None               # 验证/证伪时的迭代
    experiments: List[int] = field(default_factory=list)   # 相关实验迭代列表
    evidence: List[str] = field(default_factory=list)      # 支持/反对的证据列表
    strategy_snapshot: Optional[Dict[str, Any]] = None     # 提出时的策略快照

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'id': self.id,
            'statement': self.statement,
            'biological_basis': self.biological_basis,
            'expected_outcome': self.expected_outcome,
            'falsification_criteria': self.falsification_criteria,
            'status': self.status.value,
            'iteration_proposed': self.iteration_proposed,
            'iteration_resolved': self.iteration_resolved,
            'experiments': self.experiments,
            'evidence': self.evidence,
            'strategy_snapshot': self.strategy_snapshot,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Hypothesis':
        """从字典创建"""
        # 验证 status 值的有效性
        try:
            status = HypothesisStatus(data['status'])
        except ValueError:
            logger.warning(f"Invalid hypothesis status: {data.get('status')}, defaulting to PROPOSED")
            status = HypothesisStatus.PROPOSED

        return cls(
            id=data['id'],
            statement=data['statement'],
            biological_basis=data['biological_basis'],
            expected_outcome=data['expected_outcome'],
            falsification_criteria=data['falsification_criteria'],
            status=status,
            iteration_proposed=data.get('iteration_proposed', 0),
            iteration_resolved=data.get('iteration_resolved'),
            experiments=data.get('experiments', []),
            evidence=data.get('evidence', []),
            strategy_snapshot=data.get('strategy_snapshot'),
        )

    def format_for_prompt(self) -> str:
        """格式化为 Prompt 中的展示格式"""
        status_emoji = {
            HypothesisStatus.PROPOSED: "🔬",
            HypothesisStatus.VALIDATED: "✓",
            HypothesisStatus.INVALIDATED: "✗"
        }
        emoji = status_emoji.get(self.status, "?")

        result = f"{emoji} **{self.id}** [{self.status.value}]\n"
        result += f"   Statement: {self.statement}\n"
        result += f"   Basis: {self.biological_basis}\n"

        if self.evidence:
            result += f"   Evidence: {self.evidence[-1]}\n"

        return result


class HypothesisLedger:
    """
    假设账本 (Hypothesis Ledger)

    追踪所有科学假设的生命周期。这是 Actor 的"知识库"，
    记录了哪些生物学假设被验证、哪些被证伪。

    核心功能：
    - propose(): 提出新假设
    - validate(): 验证假设（实验成功）
    - invalidate(): 证伪假设（实验失败）
    - get_knowledge_summary(): 生成知识摘要供 LLM 参考

    使用示例：
        ledger = HypothesisLedger()

        # 提出假设
        h_id = ledger.propose(
            statement="CellMarker edges are critical for cell type prediction",
            biological_basis="Cell identity is marker-defined",
            expected_outcome="AUC > 0.85",
            falsification_criteria="If AUC < 0.80, hypothesis invalid",
            iteration=1,
            strategy={'edge_types': ['CellMarker', 'GO'], 'max_neighbors': 20}
        )

        # 实验后验证或证伪
        if experiment_successful:
            ledger.validate(h_id, iteration=2, evidence="AUC=0.87, exceeded expectation")
        else:
            ledger.invalidate(h_id, iteration=2, evidence="AUC=0.75, below threshold")

        # 获取知识摘要供下次实验参考
        summary = ledger.get_knowledge_summary()
    """

    def __init__(self):
        """初始化假设账本"""
        self.hypotheses: Dict[str, Hypothesis] = {}
        self._counter: int = 0
        self._current_hypothesis_id: Optional[str] = None  # 当前活跃假设的 ID

    def propose(
        self,
        statement: str,
        biological_basis: str,
        expected_outcome: str,
        falsification_criteria: str,
        iteration: int,
        strategy: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        提出新假设

        Args:
            statement: 假设陈述
            biological_basis: 生物学依据
            expected_outcome: 预期结果
            falsification_criteria: 证伪条件
            iteration: 当前迭代编号
            strategy: 相关策略快照

        Returns:
            str: 假设 ID (e.g., "H001")
        """
        # 验证输入不为空
        statement = statement.strip() if statement else ""
        if not statement:
            logger.warning("Empty hypothesis statement provided")
            statement = "(No statement provided)"

        self._counter += 1
        h_id = f"H{self._counter:03d}"

        hypothesis = Hypothesis(
            id=h_id,
            statement=statement,
            biological_basis=biological_basis,
            expected_outcome=expected_outcome,
            falsification_criteria=falsification_criteria,
            iteration_proposed=iteration,
            strategy_snapshot=strategy
        )

        self.hypotheses[h_id] = hypothesis
        self._current_hypothesis_id = h_id

        logger.info(f"Hypothesis proposed: {h_id} - {statement[:50]}...")
        return h_id

    def validate(self, hypothesis_id: str, iteration: int, evidence: str):
        """
        验证假设（实验成功）

        Args:
            hypothesis_id: 假设 ID
            iteration: 验证时的迭代编号
            evidence: 支持证据
        """
        h = self.hypotheses.get(hypothesis_id)
        if not h:
            logger.warning(f"Hypothesis {hypothesis_id} not found")
            return

        if h.status != HypothesisStatus.PROPOSED:
            logger.warning(f"Hypothesis {hypothesis_id} already resolved: {h.status.value}")
            return

        # 验证 evidence 不为空
        evidence = evidence.strip() if evidence else "No evidence provided"

        h.status = HypothesisStatus.VALIDATED
        h.iteration_resolved = iteration
        h.experiments.append(iteration)
        h.evidence.append(f"✓ {evidence}")

        logger.info(f"Hypothesis validated: {hypothesis_id} - {evidence}")

    def invalidate(self, hypothesis_id: str, iteration: int, evidence: str):
        """
        证伪假设（实验失败）

        Args:
            hypothesis_id: 假设 ID
            iteration: 证伪时的迭代编号
            evidence: 反对证据
        """
        h = self.hypotheses.get(hypothesis_id)
        if not h:
            logger.warning(f"Hypothesis {hypothesis_id} not found")
            return

        if h.status != HypothesisStatus.PROPOSED:
            logger.warning(f"Hypothesis {hypothesis_id} already resolved: {h.status.value}")
            return

        # 验证 evidence 不为空
        evidence = evidence.strip() if evidence else "No evidence provided"

        h.status = HypothesisStatus.INVALIDATED
        h.iteration_resolved = iteration
        h.experiments.append(iteration)
        h.evidence.append(f"✗ {evidence}")

        logger.info(f"Hypothesis invalidated: {hypothesis_id} - {evidence}")

    def get_current_hypothesis(self) -> Optional[Hypothesis]:
        """获取当前活跃的假设"""
        if self._current_hypothesis_id:
            return self.hypotheses.get(self._current_hypothesis_id)
        return None

    def get_active_hypotheses(self) -> List[Hypothesis]:
        """获取所有待验证的假设"""
        return [
            h for h in self.hypotheses.values()
            if h.status == HypothesisStatus.PROPOSED
        ]

    def get_validated_hypotheses(self) -> List[Hypothesis]:
        """获取所有已验证的假设（成功的知识）"""
        return [
            h for h in self.hypotheses.values()
            if h.status == HypothesisStatus.VALIDATED
        ]

    def get_invalidated_hypotheses(self) -> List[Hypothesis]:
        """获取所有已证伪的假设（失败的教训）"""
        return [
            h for h in self.hypotheses.values()
            if h.status == HypothesisStatus.INVALIDATED
        ]

    def get_knowledge_summary(self) -> str:
        """
        生成假设知识摘要供 LLM 参考

        包含：
        - 已验证的假设（成功的知识，应该保持）
        - 已证伪的假设（失败的教训，应该避免）
        - 当前待验证的假设

        Returns:
            str: 格式化的知识摘要
        """
        validated = self.get_validated_hypotheses()
        invalidated = self.get_invalidated_hypotheses()
        active = self.get_active_hypotheses()

        lines = ["## HYPOTHESIS LEDGER (Scientific Knowledge Base)"]

        if not self.hypotheses:
            lines.append("\nNo hypotheses recorded yet. This is the first experiment.")
            return "\n".join(lines)

        # 已验证的假设 - 成功的知识
        if validated:
            lines.append("\n### Validated Hypotheses (Proven Knowledge - BUILD ON THESE)")
            for h in validated[-3:]:  # 最近 3 个
                lines.append(f"- **{h.id}**: {h.statement}")
                lines.append(f"  Biological basis: {h.biological_basis}")
                if h.evidence:
                    lines.append(f"  Evidence: {h.evidence[-1]}")

        # 已证伪的假设 - 失败的教训
        if invalidated:
            lines.append("\n### Invalidated Hypotheses (Disproven - AVOID THESE APPROACHES)")
            for h in invalidated[-3:]:  # 最近 3 个
                lines.append(f"- **{h.id}**: {h.statement}")
                # 安全获取最后一个 evidence
                last_evidence = h.evidence[-1] if h.evidence else 'Unknown'
                lines.append(f"  Why it failed: {last_evidence}")
                # 安全获取 strategy_snapshot
                if h.strategy_snapshot:
                    edge_types = h.strategy_snapshot.get('edge_types', [])
                    lines.append(f"  Failed strategy included: edge_types={edge_types}")

        # 当前待验证的假设
        if active:
            lines.append("\n### Currently Testing")
            for h in active:
                lines.append(f"- **{h.id}**: {h.statement}")
                lines.append(f"  Falsification criteria: {h.falsification_criteria}")

        # 统计摘要
        lines.append(f"\n### Summary")
        lines.append(f"- Total hypotheses: {len(self.hypotheses)}")
        lines.append(f"- Validated: {len(validated)} | Invalidated: {len(invalidated)} | Pending: {len(active)}")

        return "\n".join(lines)

    def get_failure_patterns(self) -> str:
        """
        分析失败模式，识别应该避免的策略特征

        Returns:
            str: 失败模式分析
        """
        invalidated = self.get_invalidated_hypotheses()
        if not invalidated:
            return ""

        lines = ["## FAILURE PATTERNS (Avoid these approaches)"]

        # 收集失败策略的共同特征
        failed_edge_types: Dict[str, int] = {}
        failed_samplings: Dict[str, int] = {}

        for h in invalidated:
            if h.strategy_snapshot:
                # 统计失败的 edge_types
                for et in h.strategy_snapshot.get('edge_types', []):
                    failed_edge_types[et] = failed_edge_types.get(et, 0) + 1

                # 统计失败的 sampling
                sampling = h.strategy_snapshot.get('sampling', '')
                if sampling:
                    failed_samplings[sampling] = failed_samplings.get(sampling, 0) + 1

        if failed_edge_types:
            sorted_et = sorted(failed_edge_types.items(), key=lambda x: -x[1])
            lines.append("\nEdge types frequently in failed hypotheses:")
            for et, count in sorted_et[:3]:
                if count >= 2:
                    lines.append(f"  - {et}: appeared in {count} failed experiments")

        return "\n".join(lines) if len(lines) > 1 else ""

    def evaluate_hypothesis(
        self,
        hypothesis_id: str,
        current_metric: float,
        previous_metric: Optional[float],
        threshold: float = 0.02
    ) -> Tuple[bool, str]:
        """
        评估假设是否被验证

        Args:
            hypothesis_id: 假设 ID
            current_metric: 当前实验指标
            previous_metric: 之前的指标
            threshold: 改进阈值

        Returns:
            tuple: (is_validated, evidence_string)
        """
        h = self.hypotheses.get(hypothesis_id)
        if not h:
            return False, "Hypothesis not found"

        if previous_metric is None:
            # 第一次实验，无法比较
            return True, f"First experiment, metric={current_metric:.4f}"

        improvement = current_metric - previous_metric

        if improvement > threshold:
            return True, f"Metric improved from {previous_metric:.4f} to {current_metric:.4f} (+{improvement:.4f})"
        elif improvement < -threshold:
            return False, f"Metric decreased from {previous_metric:.4f} to {current_metric:.4f} ({improvement:.4f})"
        else:
            # 接近持平，检查是否达到预期
            return False, f"Metric stagnant at {current_metric:.4f} (delta={improvement:+.4f})"

    def save_state(self) -> Dict[str, Any]:
        """保存状态用于持久化"""
        return {
            'hypotheses': {k: v.to_dict() for k, v in self.hypotheses.items()},
            'counter': self._counter,
            'current_hypothesis_id': self._current_hypothesis_id,
        }

    def load_state(self, state: Dict[str, Any]):
        """从保存的状态恢复"""
        self.hypotheses = {
            k: Hypothesis.from_dict(v)
            for k, v in state.get('hypotheses', {}).items()
        }
        self._counter = state.get('counter', 0)
        self._current_hypothesis_id = state.get('current_hypothesis_id')

    def reset(self):
        """重置账本"""
        self.hypotheses.clear()
        self._counter = 0
        self._current_hypothesis_id = None
        logger.info("HypothesisLedger reset")
