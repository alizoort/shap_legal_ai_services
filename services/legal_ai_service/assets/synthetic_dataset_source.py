from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

RISK_LABELS: tuple[str, ...] = ("low", "medium", "high")
CATEGORY_ORDER: tuple[str, ...] = (
    "monitoring",
    "termination",
    "hiring",
    "workplace_policy",
    "discrimination",
    "compensation",
    "data_usage",
)

TRAIN_COUNTS_BY_CATEGORY: Mapping[str, int] = {
    "monitoring": 6,
    "termination": 6,
    "hiring": 6,
    "workplace_policy": 6,
    "discrimination": 6,
    "compensation": 5,
    "data_usage": 5,
}

GOLD_CATEGORIES_BY_RISK: Mapping[str, tuple[str, ...]] = {
    "low": (
        "monitoring",
        "termination",
        "hiring",
        "workplace_policy",
        "compensation",
        "data_usage",
    ),
    "medium": (
        "monitoring",
        "termination",
        "hiring",
        "workplace_policy",
        "discrimination",
        "data_usage",
    ),
    "high": (
        "monitoring",
        "termination",
        "hiring",
        "discrimination",
        "compensation",
        "data_usage",
    ),
}

_FIELD_PRIMES: tuple[int, ...] = (2, 3, 5, 7, 11, 13, 17, 19)
_TRAIN_RISK_SEEDS: Mapping[str, int] = {"low": 3, "medium": 11, "high": 19}
_GOLD_RISK_SEEDS: Mapping[str, int] = {"low": 31, "medium": 43, "high": 59}
_CATEGORY_SEEDS: Mapping[str, int] = {
    "monitoring": 2,
    "termination": 5,
    "hiring": 7,
    "workplace_policy": 11,
    "discrimination": 13,
    "compensation": 17,
    "data_usage": 19,
}


@dataclass(frozen=True)
class SyntheticRecord:
    split: str
    text: str
    risk_label: str
    rationale: str
    category: str


@dataclass(frozen=True)
class SentenceBlueprint:
    templates: tuple[str, ...]
    fields: Mapping[str, tuple[str, ...]]
    rationales: tuple[str, ...]


def build_training_records() -> list[SyntheticRecord]:
    records: list[SyntheticRecord] = []
    for risk_label in RISK_LABELS:
        for category in CATEGORY_ORDER:
            records.extend(
                _build_records_for_category(
                    split="train",
                    risk_label=risk_label,
                    category=category,
                    count=TRAIN_COUNTS_BY_CATEGORY[category],
                    base_seed=_TRAIN_RISK_SEEDS[risk_label],
                )
            )
    _validate_records(records, expected_total=120, expected_per_label=40, split="train")
    return records


def build_gold_records() -> list[SyntheticRecord]:
    records: list[SyntheticRecord] = []
    for risk_label in RISK_LABELS:
        for category in GOLD_CATEGORIES_BY_RISK[risk_label]:
            records.extend(
                _build_records_for_category(
                    split="gold",
                    risk_label=risk_label,
                    category=category,
                    count=1,
                    base_seed=_GOLD_RISK_SEEDS[risk_label],
                )
            )
    _validate_records(records, expected_total=18, expected_per_label=6, split="gold")
    return records


def build_all_records() -> list[SyntheticRecord]:
    return [*build_training_records(), *build_gold_records()]


def _build_records_for_category(
    *,
    split: str,
    risk_label: str,
    category: str,
    count: int,
    base_seed: int,
) -> list[SyntheticRecord]:
    blueprint = _BLUEPRINTS[(risk_label, category)]
    records: list[SyntheticRecord] = []
    used_texts: set[str] = set()
    field_names = tuple(blueprint.fields.keys())
    seed = base_seed + _CATEGORY_SEEDS[category]
    attempt = 0
    while len(records) < count:
        template = blueprint.templates[(seed + attempt) % len(blueprint.templates)]
        values: dict[str, str] = {}
        for index, field_name in enumerate(field_names):
            pool = blueprint.fields[field_name]
            prime = _FIELD_PRIMES[index]
            values[field_name] = pool[(seed + (attempt * prime)) % len(pool)]
        text = " ".join(template.format(**values).split())
        if text not in used_texts:
            used_texts.add(text)
            records.append(
                SyntheticRecord(
                    split=split,
                    text=text,
                    risk_label=risk_label,
                    rationale=blueprint.rationales[(seed + attempt) % len(blueprint.rationales)],
                    category=category,
                )
            )
        attempt += 1
        if attempt > count * 30:
            raise RuntimeError(
                f"Unable to build enough unique records for {split=} {risk_label=} {category=}"
            )
    return records


def _validate_records(
    records: list[SyntheticRecord],
    *,
    expected_total: int,
    expected_per_label: int,
    split: str,
) -> None:
    if len(records) != expected_total:
        raise ValueError(f"{split} dataset must contain {expected_total} rows")
    seen_texts = {record.text for record in records}
    if len(seen_texts) != len(records):
        raise ValueError(f"{split} dataset contains duplicate text rows")
    for risk_label in RISK_LABELS:
        count = sum(1 for record in records if record.risk_label == risk_label)
        if count != expected_per_label:
            raise ValueError(f"{split} dataset must contain {expected_per_label} {risk_label} rows")


_BLUEPRINTS: dict[tuple[str, str], SentenceBlueprint] = {
    (
        "high",
        "monitoring",
    ): SentenceBlueprint(
        templates=(
            "{actor} {action} {scope} {limit}.",
            "{actor} {action} {scope} {timing} {limit}.",
            "{actor} {action} {scope} {limit} {timing}.",
        ),
        fields={
            "actor": (
                "The company may",
                "Management reserves the right to",
                "Supervisors can",
                "HR may",
                "The employer may",
                "Operations leaders may",
            ),
            "action": (
                "monitor",
                "record",
                "capture logs of",
                "track",
                "review",
                "collect copies of",
            ),
            "scope": (
                "all employee communications",
                "every call, email, and chat handled by staff",
                "keystrokes and screen activity on company devices",
                "all internet browsing and messaging on work accounts",
                "every conversation sent through collaboration tools",
                "all badge swipes and application activity logs",
            ),
            "timing": (
                "at any time",
                "during and after working hours",
                "whenever management chooses",
                "without any event-based trigger",
                "even when an employee is off shift",
                "without a documented investigation",
            ),
            "limit": (
                "without notice",
                "without prior consent",
                "without telling employees why",
                "including personal conversations",
                "without limiting the review team",
                "without an approval process",
            ),
        },
        rationales=(
            "The policy allows broad monitoring without notice, consent, or clear limits.",
            "The language suggests unrestricted surveillance of employees and weak privacy safeguards.",
            "The clause authorizes expansive monitoring and does not explain transparency or boundaries.",
        ),
    ),
    (
        "medium",
        "monitoring",
    ): SentenceBlueprint(
        templates=(
            "{actor} {action} {scope} {trigger}.",
            "{actor} {action} {scope} {trigger} {gap}.",
            "{actor} {action} {scope} {gap}.",
        ),
        fields={
            "actor": (
                "Managers may",
                "The company may",
                "Supervisors can",
                "Security staff may",
                "Team leads may",
                "HR may",
            ),
            "action": (
                "review",
                "check",
                "access",
                "monitor",
                "inspect",
                "audit",
            ),
            "scope": (
                "employee email traffic",
                "badge access data",
                "camera footage from work areas",
                "device usage logs",
                "chat history on company tools",
                "location records from company devices",
            ),
            "trigger": (
                "as needed",
                "when concerns arise",
                "at management discretion",
                "when leadership believes it is appropriate",
                "for operational purposes",
                "to support business needs",
            ),
            "gap": (
                "without defining how long records are retained",
                "without explaining who approves each review",
                "without clarifying whether notice is provided",
                "without limiting the scope of collection",
                "without saying whether off-hours activity is included",
                "without listing an employee appeal path",
            ),
        },
        rationales=(
            "The monitoring clause is vague about scope, notice, and approval controls.",
            "The policy hints at monitoring but leaves key safeguards and limits undefined.",
            "The language is ambiguous enough to merit review because it lacks clear monitoring boundaries.",
        ),
    ),
    (
        "low",
        "monitoring",
    ): SentenceBlueprint(
        templates=(
            "{actor} {action} {scope} {purpose} {safeguard}.",
            "{actor} {action} {scope} {safeguard} {purpose}.",
            "{actor} {action} {scope} {purpose}, and {safeguard}.",
        ),
        fields={
            "actor": (
                "The company will",
                "Security staff will",
                "The employer will",
                "IT may",
                "Managers will",
                "The organization will",
            ),
            "action": (
                "monitor",
                "review",
                "audit",
                "check",
                "inspect",
                "access",
            ),
            "scope": (
                "company email and device activity",
                "work-system logs on company-owned equipment",
                "camera footage from public work areas",
                "network access records on employer-managed devices",
                "badge access logs for secured facilities",
                "business communications on approved channels",
            ),
            "purpose": (
                "only for security, safety, or legal investigations",
                "only for documented compliance and security reasons",
                "for limited workplace safety and cybersecurity reviews",
                "for specific investigations tied to policy breaches",
                "only when a legitimate business reason is recorded",
                "for narrowly scoped compliance checks",
            ),
            "safeguard": (
                "after providing written notice to employees",
                "with access limited to authorized reviewers",
                "while excluding private accounts and personal devices",
                "with a published retention period and approval workflow",
                "and employees are told what data may be reviewed",
                "with clear limits on timing, purpose, and access",
            ),
        },
        rationales=(
            "The monitoring language defines a legitimate purpose and clear notice-based safeguards.",
            "The clause limits scope and explains who can access monitoring data.",
            "The policy describes transparent, bounded monitoring rather than unrestricted surveillance.",
        ),
    ),
    (
        "high",
        "termination",
    ): SentenceBlueprint(
        templates=(
            "{actor} {action} {reason} {process}.",
            "{actor} {action} {process} {reason}.",
            "{actor} {action} {reason} and {process}.",
        ),
        fields={
            "actor": (
                "The company may",
                "Management can",
                "Supervisors may",
                "HR may",
                "The employer may",
                "Department heads can",
            ),
            "action": (
                "terminate employees immediately",
                "dismiss any employee on the spot",
                "end employment at once",
                "remove staff from payroll immediately",
                "fire workers without delay",
                "separate employees from service immediately",
            ),
            "reason": (
                "without notice",
                "for any reason management considers sufficient",
                "without written findings",
                "without documenting the basis for the decision",
                "without severance discussions",
                "without investigating the underlying issue",
            ),
            "process": (
                "with no appeal or review process",
                "without a hearing or response opportunity",
                "without advance notice or a corrective plan",
                "without HR approval",
                "without manager sign-off beyond verbal instruction",
                "without offering the employee a chance to respond",
            ),
        },
        rationales=(
            "The termination clause removes notice, documentation, and review safeguards.",
            "The policy authorizes immediate dismissal without due process.",
            "The language creates high employment risk because it lacks notice and fair procedure.",
        ),
    ),
    (
        "medium",
        "termination",
    ): SentenceBlueprint(
        templates=(
            "{actor} {action} {trigger} {gap}.",
            "{actor} {action} {gap} {trigger}.",
            "{actor} {action} {trigger}, while {gap}.",
        ),
        fields={
            "actor": (
                "Management may",
                "The company may",
                "Supervisors can",
                "HR may",
                "Leadership may",
                "Department managers may",
            ),
            "action": (
                "end employment",
                "separate an employee from service",
                "terminate the employment relationship",
                "dismiss an employee",
                "remove a worker from the role",
                "conclude employment",
            ),
            "trigger": (
                "at management discretion",
                "when leadership determines it is necessary",
                "when conduct is viewed as unacceptable",
                "when the business requires immediate action",
                "if expectations are not met",
                "when concerns persist",
            ),
            "gap": (
                "the policy does not define notice periods",
                "the process does not explain investigation steps",
                "the clause does not describe any review path",
                "the policy leaves documentation requirements unclear",
                "the timing of final pay is not addressed",
                "the employee response process is not described",
            ),
        },
        rationales=(
            "The termination language is discretionary and omits a clear procedure.",
            "The clause hints at a process but leaves notice, documentation, or review undefined.",
            "The wording is ambiguous enough to merit review before it is applied in practice.",
        ),
    ),
    (
        "low",
        "termination",
    ): SentenceBlueprint(
        templates=(
            "{actor} {action} {process} {notice}.",
            "{actor} {action} {notice} and {process}.",
            "{actor} {action} {process}, with {notice}.",
        ),
        fields={
            "actor": (
                "The company will",
                "Managers will",
                "HR will",
                "The employer will",
                "Leadership will",
                "Supervisors will",
            ),
            "action": (
                "handle termination decisions",
                "manage separation decisions",
                "address serious conduct issues",
                "close employment relationships",
                "process dismissals",
                "review termination recommendations",
            ),
            "process": (
                "through a documented investigation and review process",
                "using written findings, HR approval, and an employee response opportunity",
                "with a defined disciplinary path before dismissal when feasible",
                "through a policy that explains evidence review and manager sign-off",
                "with written records, escalation rules, and an appeal channel",
                "through transparent steps that define decision ownership",
            ),
            "notice": (
                "employees are informed of notice periods and final pay timelines",
                "the policy explains written notice and exceptions",
                "employees receive a chance to respond before a final decision",
                "the process includes clear communication and documented timing",
                "employees are told how the decision will be reviewed",
                "the clause sets expectations for notice and final documentation",
            ),
        },
        rationales=(
            "The termination policy describes notice, review, and documentation safeguards.",
            "The clause provides a transparent process instead of pure managerial discretion.",
            "The wording points to lower risk because fair procedure is clearly described.",
        ),
    ),
    (
        "high",
        "hiring",
    ): SentenceBlueprint(
        templates=(
            "{actor} {action} {criteria} {limit}.",
            "{actor} {action} {limit} {criteria}.",
            "{actor} {action} {criteria} and {limit}.",
        ),
        fields={
            "actor": (
                "Recruiters should",
                "The company should",
                "Hiring managers may",
                "The business will",
                "The employer should",
                "Talent staff may",
            ),
            "action": (
                "hire only candidates who are",
                "prioritize applicants who are",
                "screen out applicants who are not",
                "favor candidates who are",
                "prefer applicants who are",
                "select workers who are",
            ),
            "criteria": (
                "young and energetic",
                "male for client-facing leadership roles",
                "single and able to travel without family commitments",
                "native-born for customer trust reasons",
                "under forty for a fast-paced culture",
                "women only for front-desk presentation roles",
            ),
            "limit": (
                "even when the trait is unrelated to job duties",
                "rather than using role-based qualifications",
                "without documenting a lawful justification",
                "instead of evaluating experience and skills",
                "without any equal opportunity review",
                "as a standing hiring rule",
            ),
        },
        rationales=(
            "The hiring language uses protected or discriminatory criteria unrelated to the role.",
            "The clause explicitly prefers applicants based on protected traits.",
            "The wording creates high discrimination risk because it replaces job criteria with bias.",
        ),
    ),
    (
        "medium",
        "hiring",
    ): SentenceBlueprint(
        templates=(
            "{actor} {action} {criteria} {gap}.",
            "{actor} {action} {gap} {criteria}.",
            "{actor} {action} {criteria}, while {gap}.",
        ),
        fields={
            "actor": (
                "Recruiters may",
                "Hiring managers can",
                "The company may",
                "Talent teams may",
                "Interview panels may",
                "Leaders may",
            ),
            "action": (
                "look for candidates who fit",
                "prefer applicants who match",
                "prioritize people who reflect",
                "screen for candidates aligned with",
                "select applicants who suit",
                "favor profiles that support",
            ),
            "criteria": (
                "a youthful team culture",
                "an energetic brand image",
                "the right cultural fit",
                "a polished appearance standard",
                "a digital native mindset",
                "a strong personality match",
            ),
            "gap": (
                "without defining objective job criteria",
                "without explaining how bias is controlled",
                "without listing role-based scoring rules",
                "without clarifying equal opportunity safeguards",
                "without describing how interviews are standardized",
                "without documenting lawful business necessity",
            ),
        },
        rationales=(
            "The hiring language is not overtly discriminatory but relies on subjective criteria.",
            "The policy uses vague hiring preferences that could hide biased decision-making.",
            "The wording merits review because role-based safeguards are not clearly defined.",
        ),
    ),
    (
        "low",
        "hiring",
    ): SentenceBlueprint(
        templates=(
            "{actor} {action} {criteria} {safeguard}.",
            "{actor} {action} {safeguard} {criteria}.",
            "{actor} {action} {criteria}, with {safeguard}.",
        ),
        fields={
            "actor": (
                "Hiring managers will",
                "Recruiters will",
                "The company will",
                "Interview panels will",
                "Talent teams will",
                "The employer will",
            ),
            "action": (
                "evaluate applicants using",
                "select candidates through",
                "screen applicants against",
                "make hiring decisions with",
                "rank candidates based on",
                "assess applicants through",
            ),
            "criteria": (
                "job-related qualifications and documented competencies",
                "role-specific experience, skills, and interview scoring",
                "objective criteria tied to the position description",
                "standardized interview questions and relevant credentials",
                "written scorecards linked to business needs",
                "published role requirements and structured assessments",
            ),
            "safeguard": (
                "an equal opportunity statement is included in the process",
                "protected traits are excluded from decision criteria",
                "interviewers receive bias-mitigation guidance",
                "decisions are documented and reviewed for consistency",
                "the process uses standardized scoring across candidates",
                "the policy explains how fair hiring is monitored",
            ),
        },
        rationales=(
            "The hiring policy centers on objective, role-based criteria and fairness controls.",
            "The clause describes a non-discriminatory hiring process with structured safeguards.",
            "The wording points to low risk because selection is tied to job requirements.",
        ),
    ),
    (
        "high",
        "workplace_policy",
    ): SentenceBlueprint(
        templates=(
            "{actor} {action} {scope} {limit}.",
            "{actor} {action} {limit} {scope}.",
            "{actor} {action} {scope} and {limit}.",
        ),
        fields={
            "actor": (
                "Management may",
                "The company can",
                "Supervisors may",
                "HR may",
                "Department leaders can",
                "The employer may",
            ),
            "action": (
                "change workplace rules",
                "discipline employees",
                "issue final warnings",
                "impose sanctions",
                "move employees to unpaid leave",
                "suspend staff",
            ),
            "scope": (
                "whenever leadership believes it is appropriate",
                "based on verbal direction alone",
                "without publishing a written process",
                "without explaining the standards applied",
                "without reviewing past practice",
                "without a documented complaint or incident record",
            ),
            "limit": (
                "and employees have no review path",
                "without advance notice",
                "without written justification",
                "without consistency checks across teams",
                "without HR approval",
                "without a chance for the employee to respond",
            ),
        },
        rationales=(
            "The workplace policy grants broad disciplinary discretion without transparent procedure.",
            "The clause allows sanctions without notice, documentation, or review safeguards.",
            "The language is high risk because it replaces policy clarity with unilateral discretion.",
        ),
    ),
    (
        "medium",
        "workplace_policy",
    ): SentenceBlueprint(
        templates=(
            "{actor} {action} {scope} {gap}.",
            "{actor} {action} {gap} {scope}.",
            "{actor} {action} {scope}, but {gap}.",
        ),
        fields={
            "actor": (
                "Managers may",
                "The company may",
                "HR can",
                "Supervisors may",
                "Leadership may",
                "Policy owners may",
            ),
            "action": (
                "apply workplace rules",
                "take disciplinary action",
                "address conduct issues",
                "issue corrective measures",
                "update work standards",
                "respond to employee concerns",
            ),
            "scope": (
                "as appropriate to the circumstances",
                "at management discretion",
                "when leadership sees a business need",
                "using case-by-case judgment",
                "based on operational needs",
                "whenever a manager believes action is necessary",
            ),
            "gap": (
                "the policy does not define escalation steps",
                "the complaint review route is not clearly described",
                "the document does not explain notice expectations",
                "the approval chain for discipline is unclear",
                "the rule changes are not tied to a written update process",
                "the employee response process is left vague",
            ),
        },
        rationales=(
            "The workplace policy is partially structured but still too vague on procedure.",
            "The clause uses discretionary language and leaves key process safeguards unclear.",
            "The wording warrants review because transparency and escalation rules are incomplete.",
        ),
    ),
    (
        "low",
        "workplace_policy",
    ): SentenceBlueprint(
        templates=(
            "{actor} {action} {scope} {safeguard}.",
            "{actor} {action} {safeguard} {scope}.",
            "{actor} {action} {scope}, and {safeguard}.",
        ),
        fields={
            "actor": (
                "The company will",
                "Managers will",
                "HR will",
                "Leadership will",
                "Policy owners will",
                "Supervisors will",
            ),
            "action": (
                "apply workplace rules",
                "manage conduct issues",
                "handle complaints",
                "update work standards",
                "address policy breaches",
                "review disciplinary matters",
            ),
            "scope": (
                "through a written policy available to employees",
                "using a defined complaint and investigation process",
                "through documented escalation and review steps",
                "under standards that explain expected conduct",
                "with written criteria for warnings and sanctions",
                "through a process that identifies decision-makers",
            ),
            "safeguard": (
                "employees are told how to raise concerns and seek review",
                "the policy explains notice, documentation, and appeal options",
                "changes are published before they take effect",
                "discipline decisions require written records and approvals",
                "the process defines timelines and communication expectations",
                "employees have a clear route to respond and escalate",
            ),
        },
        rationales=(
            "The workplace policy explains expectations, review paths, and documentation controls.",
            "The clause is transparent about how rules are applied and challenged.",
            "The wording points to low risk because employee process protections are stated clearly.",
        ),
    ),
    (
        "high",
        "discrimination",
    ): SentenceBlueprint(
        templates=(
            "{actor} {action} {criteria} {limit}.",
            "{actor} {action} {limit} {criteria}.",
            "{actor} {action} {criteria} and {limit}.",
        ),
        fields={
            "actor": (
                "Managers may",
                "The company may",
                "Supervisors can",
                "The employer may",
                "Department leaders can",
                "HR may",
            ),
            "action": (
                "assign work based on",
                "deny opportunities to employees because of",
                "reserve promotions for staff with",
                "screen employees according to",
                "limit travel opportunities based on",
                "choose leaders according to",
            ),
            "criteria": (
                "gender expectations",
                "age and appearance",
                "marital status",
                "national origin",
                "pregnancy status",
                "religious background",
            ),
            "limit": (
                "rather than job performance",
                "without any legitimate business justification",
                "as a standing management preference",
                "even when the factor is unrelated to the role",
                "without an equal opportunity review",
                "instead of using objective performance criteria",
            ),
        },
        rationales=(
            "The policy uses protected traits in employment decisions unrelated to job duties.",
            "The clause directly creates discrimination risk by relying on protected characteristics.",
            "The wording is high risk because opportunity decisions are tied to biased criteria.",
        ),
    ),
    (
        "medium",
        "discrimination",
    ): SentenceBlueprint(
        templates=(
            "{actor} {action} {criteria} {gap}.",
            "{actor} {action} {gap} {criteria}.",
            "{actor} {action} {criteria}, while {gap}.",
        ),
        fields={
            "actor": (
                "Managers may",
                "Leadership can",
                "The company may",
                "Supervisors may",
                "HR can",
                "Department heads may",
            ),
            "action": (
                "consider",
                "weigh",
                "take into account",
                "factor in",
                "reference",
                "look at",
            ),
            "criteria": (
                "team fit and presentation style",
                "client preference signals",
                "appearance expectations",
                "perceived cultural alignment",
                "communication style assumptions",
                "personality matching",
            ),
            "gap": (
                "without defining anti-bias controls",
                "without showing how subjective judgments are checked",
                "without clarifying what job criteria outweigh personal preferences",
                "without documenting fairness review steps",
                "without saying how protected traits are excluded",
                "without a standard scoring framework",
            ),
        },
        rationales=(
            "The policy does not state overt bias, but it relies on subjective judgments that can mask discrimination.",
            "The wording leaves too much room for personal preference without anti-bias controls.",
            "The clause should be reviewed because subjective employment criteria are not bounded clearly.",
        ),
    ),
    (
        "low",
        "discrimination",
    ): SentenceBlueprint(
        templates=(
            "{actor} {action} {criteria} {safeguard}.",
            "{actor} {action} {safeguard} {criteria}.",
            "{actor} {action} {criteria}, and {safeguard}.",
        ),
        fields={
            "actor": (
                "The company will",
                "Managers will",
                "HR will",
                "Leadership will",
                "Supervisors will",
                "Policy owners will",
            ),
            "action": (
                "make employment decisions using",
                "assess employees through",
                "handle promotions with",
                "allocate opportunities according to",
                "review employment decisions with",
                "evaluate candidates and staff against",
            ),
            "criteria": (
                "documented, role-related criteria",
                "objective performance and conduct standards",
                "job-based qualifications and measurable expectations",
                "written competency and performance indicators",
                "standardized decision criteria tied to the role",
                "published performance factors rather than personal traits",
            ),
            "safeguard": (
                "protected traits are excluded from the decision process",
                "the policy includes equal opportunity and anti-retaliation safeguards",
                "decision records are reviewed for consistency and fairness",
                "employees can raise concerns through a defined review path",
                "the process documents how bias is mitigated",
                "managers are instructed not to use personal characteristics",
            ),
        },
        rationales=(
            "The employment policy ties decisions to objective criteria and excludes protected traits.",
            "The clause describes clear anti-discrimination safeguards and review controls.",
            "The wording points to lower risk because fairness standards are explicit and documented.",
        ),
    ),
    (
        "high",
        "compensation",
    ): SentenceBlueprint(
        templates=(
            "{actor} {action} {change} {limit}.",
            "{actor} {action} {limit} {change}.",
            "{actor} {action} {change} and {limit}.",
        ),
        fields={
            "actor": (
                "Management may",
                "The company can",
                "Supervisors may",
                "The employer may",
                "HR may",
                "Finance leaders may",
            ),
            "action": (
                "reduce pay",
                "deduct wages",
                "withhold bonuses",
                "change salary terms",
                "cut compensation",
                "remove overtime payments",
            ),
            "change": (
                "without employee agreement",
                "without advance notice",
                "for business reasons alone",
                "when managers believe it is necessary",
                "without a written amendment",
                "without recording the reason for the change",
            ),
            "limit": (
                "and employees are expected to continue working",
                "without describing any appeal process",
                "without clarifying local wage-law compliance",
                "without documenting consent or acknowledgment",
                "even if the employee already performed the work",
                "without explaining how final pay is handled",
            ),
        },
        rationales=(
            "The compensation clause permits unilateral pay changes without consent or process.",
            "The policy creates wage and employment risk by allowing deductions or cuts without safeguards.",
            "The wording is high risk because compensation changes are not tied to consent or compliance controls.",
        ),
    ),
    (
        "medium",
        "compensation",
    ): SentenceBlueprint(
        templates=(
            "{actor} {action} {change} {gap}.",
            "{actor} {action} {gap} {change}.",
            "{actor} {action} {change}, but {gap}.",
        ),
        fields={
            "actor": (
                "The company may",
                "Managers may",
                "Leadership can",
                "HR may",
                "Finance may",
                "Supervisors may",
            ),
            "action": (
                "adjust compensation",
                "change pay terms",
                "modify incentive plans",
                "review salary arrangements",
                "revise overtime practices",
                "update pay structures",
            ),
            "change": (
                "as business needs require",
                "when conditions change",
                "at management discretion",
                "for operational reasons",
                "based on performance concerns",
                "if costs increase",
            ),
            "gap": (
                "the policy does not define notice periods",
                "the policy does not explain employee acknowledgment steps",
                "the legal review process is not described",
                "the document leaves consent requirements unclear",
                "the timing of pay changes is not specified",
                "the process for handling disputes is not explained",
            ),
        },
        rationales=(
            "The compensation language is vague and leaves notice or consent expectations unclear.",
            "The policy hints at pay changes but does not clearly document safeguards.",
            "The wording warrants review because compensation changes are described too loosely.",
        ),
    ),
    (
        "low",
        "compensation",
    ): SentenceBlueprint(
        templates=(
            "{actor} {action} {change} {safeguard}.",
            "{actor} {action} {safeguard} {change}.",
            "{actor} {action} {change}, and {safeguard}.",
        ),
        fields={
            "actor": (
                "The company will",
                "Managers will",
                "HR will",
                "Finance will",
                "Leadership will",
                "The employer will",
            ),
            "action": (
                "manage compensation changes",
                "review pay adjustments",
                "handle incentive plan updates",
                "apply salary changes",
                "address overtime corrections",
                "communicate compensation decisions",
            ),
            "change": (
                "through written notice and documented approval",
                "with employee acknowledgment when terms change",
                "using a process that explains timing and reason",
                "through published payroll and wage-compliance rules",
                "with role-based approval and written records",
                "through a documented process reviewed by HR and finance",
            ),
            "safeguard": (
                "the policy explains how lawful deductions and final pay are handled",
                "employees are told when changes take effect and why",
                "the process requires compliance review before implementation",
                "the policy states when consent or acknowledgment is needed",
                "employees can raise pay concerns through a defined channel",
                "the clause separates payroll corrections from disciplinary action",
            ),
        },
        rationales=(
            "The compensation policy explains notice, approval, and compliance safeguards.",
            "The clause is transparent about when pay changes can occur and how they are documented.",
            "The wording points to lower risk because pay practices are structured and reviewable.",
        ),
    ),
    (
        "high",
        "data_usage",
    ): SentenceBlueprint(
        templates=(
            "{actor} {action} {scope} {limit}.",
            "{actor} {action} {limit} {scope}.",
            "{actor} {action} {scope} and {limit}.",
        ),
        fields={
            "actor": (
                "The company may",
                "Management can",
                "HR may",
                "The employer may",
                "Security teams may",
                "Operations may",
            ),
            "action": (
                "use",
                "share",
                "collect",
                "store",
                "transfer",
                "combine",
            ),
            "scope": (
                "employee location data outside working hours",
                "personal phone numbers and family contacts for any internal purpose",
                "biometric scans and health-related information without a limited use case",
                "private device data captured through work applications",
                "personal data with external vendors without a narrow business reason",
                "all employee records in a single unrestricted internal database",
            ),
            "limit": (
                "without consent",
                "without telling employees how the data will be used",
                "without a retention limit",
                "without restricting who can access the information",
                "without documenting a lawful basis",
                "without any opt-out or review mechanism",
            ),
        },
        rationales=(
            "The data-use clause allows broad collection or sharing without consent or purpose limits.",
            "The policy creates privacy and employment risk through unrestricted employee data usage.",
            "The wording is high risk because sensitive employee data can be used without clear controls.",
        ),
    ),
    (
        "medium",
        "data_usage",
    ): SentenceBlueprint(
        templates=(
            "{actor} {action} {scope} {gap}.",
            "{actor} {action} {gap} {scope}.",
            "{actor} {action} {scope}, while {gap}.",
        ),
        fields={
            "actor": (
                "The company may",
                "HR may",
                "Managers can",
                "Security teams may",
                "The employer may",
                "Operations may",
            ),
            "action": (
                "retain",
                "review",
                "use",
                "share",
                "access",
                "combine",
            ),
            "scope": (
                "employee contact and device records",
                "badge and attendance data",
                "location and access logs",
                "performance-related personal information",
                "training and HR records",
                "workplace investigation files",
            ),
            "gap": (
                "without clearly stating the retention period",
                "without explaining the consent mechanism",
                "without listing which teams can access the data",
                "without defining the business purpose precisely",
                "without clarifying vendor-sharing limits",
                "without telling employees how corrections can be requested",
            ),
        },
        rationales=(
            "The data-use language suggests a business purpose but leaves key privacy safeguards unclear.",
            "The clause is ambiguous about consent, retention, or access controls.",
            "The wording should be reviewed because employee data handling rules are incomplete.",
        ),
    ),
    (
        "low",
        "data_usage",
    ): SentenceBlueprint(
        templates=(
            "{actor} {action} {scope} {safeguard}.",
            "{actor} {action} {safeguard} {scope}.",
            "{actor} {action} {scope}, and {safeguard}.",
        ),
        fields={
            "actor": (
                "The company will",
                "HR will",
                "Security teams will",
                "The employer will",
                "Managers will",
                "The organization will",
            ),
            "action": (
                "use",
                "retain",
                "review",
                "access",
                "process",
                "handle",
            ),
            "scope": (
                "employee data for defined HR, payroll, security, or legal purposes",
                "workplace records only for documented business needs",
                "personal information according to a published privacy notice",
                "employee records through role-based access controls",
                "sensitive HR data only when the stated purpose requires it",
                "employee information using purpose-limited workflows",
            ),
            "safeguard": (
                "employees are informed about retention, access, and correction rights",
                "the policy requires notice, purpose limitation, and restricted access",
                "consent or acknowledgment is obtained when required",
                "vendor sharing is limited and documented",
                "the process explains how long data is stored and who may review it",
                "employees are told how their information is collected and used",
            ),
        },
        rationales=(
            "The data-use clause explains purpose, notice, and access limitations.",
            "The policy points to low risk because employee data handling is transparent and constrained.",
            "The wording includes consent, notice, or access safeguards that reduce compliance risk.",
        ),
    ),
}

