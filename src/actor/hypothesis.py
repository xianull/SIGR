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
from typing import List, Optional, Dict, Any, Tuple, Set

logger = logging.getLogger(__name__)


# =============================================================================
# 边类型贡献追踪 (Edge Type Contribution Tracking)
# =============================================================================

@dataclass
class EdgeTypeContribution:
    """
    追踪每种边类型对指标的边际贡献 (Marginal Contribution Tracking)

    用于实现"消融实验"逻辑：
    - 记录添加/移除某边类型时指标的变化
    - 帮助 Actor 识别哪些边类型是噪声

    Attributes:
        edge_type: 边类型名称 (e.g., "PPI", "HPO", "Reactome")
        times_added: 该边类型被添加的次数
        times_removed: 该边类型被移除的次数
        avg_delta_when_added: 添加时的平均指标变化 (EMA)
        avg_delta_when_removed: 移除时的平均指标变化 (EMA)
        last_metric_with: 最近一次包含该边类型时的指标
        last_metric_without: 最近一次不包含该边类型时的指标
    """
    edge_type: str
    times_added: int = 0
    times_removed: int = 0
    avg_delta_when_added: float = 0.0
    avg_delta_when_removed: float = 0.0
    last_metric_with: float = 0.0
    last_metric_without: float = 0.0

    # EMA 衰减系数
    EMA_DECAY: float = 0.7

    def record_addition(self, metric_delta: float, current_metric: float):
        """记录边类型被添加时的效果"""
        self.times_added += 1
        if self.times_added == 1:
            self.avg_delta_when_added = metric_delta
        else:
            self.avg_delta_when_added = (
                self.EMA_DECAY * self.avg_delta_when_added +
                (1 - self.EMA_DECAY) * metric_delta
            )
        self.last_metric_with = current_metric

    def record_removal(self, metric_delta: float, current_metric: float):
        """记录边类型被移除时的效果"""
        self.times_removed += 1
        if self.times_removed == 1:
            self.avg_delta_when_removed = metric_delta
        else:
            self.avg_delta_when_removed = (
                self.EMA_DECAY * self.avg_delta_when_removed +
                (1 - self.EMA_DECAY) * metric_delta
            )
        self.last_metric_without = current_metric

    def get_net_contribution(self) -> float:
        """
        计算净贡献：添加时的效果 - 移除时的效果

        正值 = 添加有益，移除有害 = BENEFICIAL
        负值 = 添加有害，移除有益 = NOISE
        接近零 = NEUTRAL
        """
        return self.avg_delta_when_added - self.avg_delta_when_removed

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'edge_type': self.edge_type,
            'times_added': self.times_added,
            'times_removed': self.times_removed,
            'avg_delta_when_added': self.avg_delta_when_added,
            'avg_delta_when_removed': self.avg_delta_when_removed,
            'last_metric_with': self.last_metric_with,
            'last_metric_without': self.last_metric_without,
            'net_contribution': self.get_net_contribution(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EdgeTypeContribution':
        """从字典创建"""
        return cls(
            edge_type=data['edge_type'],
            times_added=data.get('times_added', 0),
            times_removed=data.get('times_removed', 0),
            avg_delta_when_added=data.get('avg_delta_when_added', 0.0),
            avg_delta_when_removed=data.get('avg_delta_when_removed', 0.0),
            last_metric_with=data.get('last_metric_with', 0.0),
            last_metric_without=data.get('last_metric_without', 0.0),
        )


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

        # 边类型贡献追踪 (Edge Type Contribution Tracking)
        self.edge_contributions: Dict[str, EdgeTypeContribution] = {}
        self._prev_edge_types: Optional[Set[str]] = None
        self._prev_metric: Optional[float] = None
        self._best_metric_strategy_neighbors: Optional[int] = None  # 最佳指标时的 max_neighbors

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
            'edge_contributions': {k: v.to_dict() for k, v in self.edge_contributions.items()},
            'prev_edge_types': list(self._prev_edge_types) if self._prev_edge_types else None,
            'prev_metric': self._prev_metric,
            'best_metric_strategy_neighbors': self._best_metric_strategy_neighbors,
        }

    def load_state(self, state: Dict[str, Any]):
        """从保存的状态恢复"""
        self.hypotheses = {
            k: Hypothesis.from_dict(v)
            for k, v in state.get('hypotheses', {}).items()
        }
        self._counter = state.get('counter', 0)
        self._current_hypothesis_id = state.get('current_hypothesis_id')
        # 恢复边类型贡献追踪
        self.edge_contributions = {
            k: EdgeTypeContribution.from_dict(v)
            for k, v in state.get('edge_contributions', {}).items()
        }
        prev_et = state.get('prev_edge_types')
        self._prev_edge_types = set(prev_et) if prev_et else None
        self._prev_metric = state.get('prev_metric')
        self._best_metric_strategy_neighbors = state.get('best_metric_strategy_neighbors')

    def reset(self):
        """重置账本"""
        self.hypotheses.clear()
        self._counter = 0
        self._current_hypothesis_id = None
        # 重置边类型贡献追踪
        self.edge_contributions.clear()
        self._prev_edge_types = None
        self._prev_metric = None
        self._best_metric_strategy_neighbors = None
        logger.info("HypothesisLedger reset")

    # =========================================================================
    # 边类型贡献追踪方法 (Edge Type Contribution Tracking Methods)
    # =========================================================================

    def record_edge_contribution(
        self,
        current_edge_types: List[str],
        current_metric: float,
        iteration: int,
        max_neighbors: Optional[int] = None,
    ):
        """
        记录边类型变化的边际贡献

        通过比较本次和上次的边类型集合，追踪：
        - 哪些边类型被添加，以及添加后指标变化
        - 哪些边类型被移除，以及移除后指标变化

        Args:
            current_edge_types: 当前使用的边类型列表
            current_metric: 当前实验的指标值
            iteration: 当前迭代编号
            max_neighbors: 当前策略的 max_neighbors 值
        """
        curr_set = set(current_edge_types)

        # 首次调用，初始化状态
        if self._prev_edge_types is None or self._prev_metric is None:
            self._prev_edge_types = curr_set
            self._prev_metric = current_metric
            if max_neighbors:
                self._best_metric_strategy_neighbors = max_neighbors
            logger.debug(f"Edge contribution tracking initialized: edges={curr_set}")
            return

        # 计算指标变化
        metric_delta = current_metric - self._prev_metric

        # 识别添加和移除的边类型
        added_edges = curr_set - self._prev_edge_types
        removed_edges = self._prev_edge_types - curr_set

        # 记录添加的边类型效果
        for edge_type in added_edges:
            if edge_type not in self.edge_contributions:
                self.edge_contributions[edge_type] = EdgeTypeContribution(edge_type=edge_type)
            self.edge_contributions[edge_type].record_addition(metric_delta, current_metric)
            logger.debug(
                f"Edge {edge_type} added: delta={metric_delta:+.4f}, "
                f"avg_when_added={self.edge_contributions[edge_type].avg_delta_when_added:+.4f}"
            )

        # 记录移除的边类型效果
        for edge_type in removed_edges:
            if edge_type not in self.edge_contributions:
                self.edge_contributions[edge_type] = EdgeTypeContribution(edge_type=edge_type)
            self.edge_contributions[edge_type].record_removal(metric_delta, current_metric)
            logger.debug(
                f"Edge {edge_type} removed: delta={metric_delta:+.4f}, "
                f"avg_when_removed={self.edge_contributions[edge_type].avg_delta_when_removed:+.4f}"
            )

        # 更新最佳指标时的 max_neighbors
        if max_neighbors and (
            self._best_metric_strategy_neighbors is None or
            current_metric > self._prev_metric
        ):
            self._best_metric_strategy_neighbors = max_neighbors

        # 更新状态
        self._prev_edge_types = curr_set
        self._prev_metric = current_metric

    def get_edge_contribution_summary(self) -> str:
        """
        生成边类型贡献摘要供 Bio-CoT 使用

        格式化输出各边类型的边际贡献，帮助 Actor 识别噪声边类型。

        Returns:
            str: 格式化的边类型贡献摘要
        """
        if not self.edge_contributions:
            return ""

        lines = ["## EDGE TYPE ABLATION HISTORY (Marginal Contributions)"]
        lines.append("Based on historical experiments, here's how each edge type affected performance:")
        lines.append("")

        # 按净贡献排序
        sorted_contributions = sorted(
            self.edge_contributions.values(),
            key=lambda x: x.get_net_contribution(),
            reverse=True
        )

        for contrib in sorted_contributions:
            # 只显示有足够样本的边类型
            total_samples = contrib.times_added + contrib.times_removed
            if total_samples < 2:
                continue

            classification = self._classify_edge_type(contrib)
            net_contrib = contrib.get_net_contribution()

            lines.append(
                f"- **{contrib.edge_type}**: "
                f"added={contrib.avg_delta_when_added:+.4f} (n={contrib.times_added}), "
                f"removed={contrib.avg_delta_when_removed:+.4f} (n={contrib.times_removed}) "
                f"-> **{classification}** (net={net_contrib:+.4f})"
            )

        if len(lines) <= 2:
            return ""

        # 添加解读指南
        lines.append("")
        lines.append("**Interpretation Guide:**")
        lines.append("- BENEFICIAL: Adding improves performance, removing hurts - KEEP this edge type")
        lines.append("- LIKELY NOISE: Adding hurts performance, removing helps - CONSIDER REMOVING")
        lines.append("- NEUTRAL: No clear pattern - experiment further")

        return "\n".join(lines)

    def _classify_edge_type(self, contrib: EdgeTypeContribution) -> str:
        """
        将边类型分类为 BENEFICIAL / LIKELY NOISE / NEUTRAL

        分类逻辑：
        - BENEFICIAL: 添加时指标提升，移除时指标下降
        - LIKELY NOISE: 添加时指标下降，移除时指标提升
        - NEUTRAL: 效果不明显或矛盾

        Args:
            contrib: EdgeTypeContribution 实例

        Returns:
            str: 分类标签
        """
        add_effect = contrib.avg_delta_when_added
        remove_effect = contrib.avg_delta_when_removed

        # 添加有益（>1%）且移除有害（<0）
        if add_effect > 0.01 and remove_effect < 0:
            return "BENEFICIAL"

        # 添加有害（<-1%）且移除有益（>0）
        if add_effect < -0.01 and remove_effect > 0:
            return "LIKELY NOISE"

        # 添加有害但移除也有害或无效
        if add_effect < -0.01:
            return "POSSIBLY NOISE"

        # 添加有益但移除也有益（矛盾）
        if add_effect > 0.01 and remove_effect > 0:
            return "UNCERTAIN"

        # 效果都很小
        if abs(add_effect) < 0.005 and abs(remove_effect) < 0.005:
            return "NEUTRAL"

        return "UNCERTAIN"

    def get_noisy_edge_types(self, threshold: float = -0.01) -> List[str]:
        """
        获取可能是噪声的边类型列表

        Args:
            threshold: 净贡献低于此值认为是噪声

        Returns:
            List[str]: 噪声边类型列表
        """
        noisy = []
        for edge_type, contrib in self.edge_contributions.items():
            if contrib.times_added + contrib.times_removed < 2:
                continue
            if contrib.get_net_contribution() < threshold:
                noisy.append(edge_type)
        return noisy

    def get_beneficial_edge_types(self, threshold: float = 0.01) -> List[str]:
        """
        获取有益的边类型列表

        Args:
            threshold: 净贡献高于此值认为是有益

        Returns:
            List[str]: 有益边类型列表
        """
        beneficial = []
        for edge_type, contrib in self.edge_contributions.items():
            if contrib.times_added + contrib.times_removed < 2:
                continue
            if contrib.get_net_contribution() > threshold:
                beneficial.append(edge_type)
        return beneficial

    def should_prune_context(self, current_max_neighbors: int) -> bool:
        """
        判断是否应该进入 PRUNE 模式

        条件：当前 max_neighbors 超过最佳指标时的 1.5 倍

        Args:
            current_max_neighbors: 当前策略的 max_neighbors

        Returns:
            bool: 是否应该剪枝
        """
        if self._best_metric_strategy_neighbors is None:
            return False
        return current_max_neighbors > self._best_metric_strategy_neighbors * 1.5
