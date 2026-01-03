"""
Compliance & Audit Module - Phase 6: Enterprise Integration
Regulatory compliance and audit logging for enterprise fraud detection
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import hashlib
import uuid
import os
from enum import Enum

logger = logging.getLogger(__name__)

class ComplianceStandard(Enum):
    """Supported compliance standards."""
    GDPR = "gdpr"
    PCI_DSS = "pci_dss"
    SOX = "sox"
    HIPAA = "hipaa"
    CCPA = "ccpa"

class AuditEventType(Enum):
    """Types of audit events."""
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    FRAUD_DETECTION = "fraud_detection"
    CONFIGURATION_CHANGE = "configuration_change"
    SYSTEM_ACCESS = "system_access"

@dataclass
class AuditLogEntry:
    """Audit log entry for compliance."""
    id: str
    timestamp: datetime
    event_type: str
    user_id: Optional[str]
    session_id: Optional[str]
    ip_address: str
    user_agent: str
    resource: str
    action: str
    status: str
    details: Dict[str, Any]
    compliance_tags: List[str]
    risk_score: float
    data_classification: str

@dataclass
class DataRetentionPolicy:
    """Data retention policy configuration."""
    data_type: str
    retention_period_days: int
    compliance_standard: str
    encryption_required: bool
    audit_required: bool

@dataclass
class ComplianceCheck:
    """Compliance check result."""
    check_id: str
    standard: str
    check_type: str
    status: str
    timestamp: datetime
    details: Dict[str, Any]
    remediation_required: bool

class AuditLogger:
    """Comprehensive audit logging for compliance."""

    def __init__(self, log_file: str = "audit.log", max_entries: int = 100000):
        self.log_file = log_file
        self.max_entries = max_entries
        self.entries: List[AuditLogEntry] = []
        self._load_existing_logs()

    def log_event(self, event_type: AuditEventType, user_id: Optional[str],
                  session_id: Optional[str], ip_address: str, user_agent: str,
                  resource: str, action: str, status: str,
                  details: Dict[str, Any], compliance_tags: List[str] = None,
                  risk_score: float = 0.0, data_classification: str = "internal") -> str:
        """Log an audit event."""

        entry = AuditLogEntry(
            id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            event_type=event_type.value,
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource=resource,
            action=action,
            status=status,
            details=details,
            compliance_tags=compliance_tags or [],
            risk_score=risk_score,
            data_classification=data_classification
        )

        self.entries.append(entry)

        # Maintain max entries limit
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]

        # Write to file
        self._write_entry(entry)

        logger.info(f"Audit event logged: {event_type.value} - {action} on {resource}")
        return entry.id

    def get_entries(self, user_id: Optional[str] = None,
                   event_type: Optional[str] = None,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   limit: int = 100) -> List[AuditLogEntry]:
        """Retrieve audit entries with filtering."""

        filtered_entries = self.entries

        if user_id:
            filtered_entries = [e for e in filtered_entries if e.user_id == user_id]

        if event_type:
            filtered_entries = [e for e in filtered_entries if e.event_type == event_type]

        if start_time:
            filtered_entries = [e for e in filtered_entries if e.timestamp >= start_time]

        if end_time:
            filtered_entries = [e for e in filtered_entries if e.timestamp <= end_time]

        return filtered_entries[-limit:]

    def get_compliance_report(self, standard: ComplianceStandard,
                            start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Generate compliance report for a specific standard."""

        relevant_entries = [
            e for e in self.entries
            if standard.value in e.compliance_tags
            and start_time <= e.timestamp <= end_time
        ]

        violations = [e for e in relevant_entries if e.status == "failure" or e.risk_score > 0.7]

        return {
            "standard": standard.value,
            "period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "total_events": len(relevant_entries),
            "violations": len(violations),
            "compliance_rate": (len(relevant_entries) - len(violations)) / max(len(relevant_entries), 1),
            "high_risk_events": len([e for e in relevant_entries if e.risk_score > 0.8]),
            "events_by_type": self._group_events_by_type(relevant_entries)
        }

    def _group_events_by_type(self, entries: List[AuditLogEntry]) -> Dict[str, int]:
        """Group events by type."""
        type_counts = {}
        for entry in entries:
            type_counts[entry.event_type] = type_counts.get(entry.event_type, 0) + 1
        return type_counts

    def _write_entry(self, entry: AuditLogEntry):
        """Write audit entry to file."""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                json.dump(asdict(entry), f, default=str, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            logger.error(f"Failed to write audit entry to file: {e}")

    def _load_existing_logs(self):
        """Load existing audit logs from file."""
        if not os.path.exists(self.log_file):
            return

        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        # Convert timestamp string back to datetime
                        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                        entry = AuditLogEntry(**data)
                        self.entries.append(entry)
                    except Exception as e:
                        logger.warning(f"Failed to parse audit log line: {e}")
        except Exception as e:
            logger.error(f"Failed to load audit logs: {e}")

class DataRetentionManager:
    """Manages data retention according to compliance policies."""

    def __init__(self):
        self.policies: Dict[str, DataRetentionPolicy] = {}
        self._setup_default_policies()

    def add_policy(self, policy: DataRetentionPolicy):
        """Add a data retention policy."""
        self.policies[policy.data_type] = policy
        logger.info(f"Added retention policy for {policy.data_type}")

    def get_policy(self, data_type: str) -> Optional[DataRetentionPolicy]:
        """Get retention policy for data type."""
        return self.policies.get(data_type)

    def should_retain_data(self, data_type: str, created_at: datetime) -> bool:
        """Check if data should be retained based on policy."""
        policy = self.get_policy(data_type)
        if not policy:
            return True  # Retain by default if no policy

        retention_cutoff = datetime.utcnow() - timedelta(days=policy.retention_period_days)
        return created_at > retention_cutoff

    def get_data_to_purge(self, data_type: str, current_data: List[Dict[str, Any]],
                         timestamp_field: str = 'timestamp') -> List[Dict[str, Any]]:
        """Get data that should be purged based on retention policy."""
        policy = self.get_policy(data_type)
        if not policy:
            return []

        retention_cutoff = datetime.utcnow() - timedelta(days=policy.retention_period_days)

        to_purge = []
        for item in current_data:
            item_timestamp = item.get(timestamp_field)
            if isinstance(item_timestamp, str):
                item_timestamp = datetime.fromisoformat(item_timestamp)

            if item_timestamp and item_timestamp < retention_cutoff:
                to_purge.append(item)

        return to_purge

    def _setup_default_policies(self):
        """Setup default retention policies."""
        default_policies = [
            DataRetentionPolicy(
                data_type="transaction_data",
                retention_period_days=2555,  # 7 years for PCI DSS
                compliance_standard="pci_dss",
                encryption_required=True,
                audit_required=True
            ),
            DataRetentionPolicy(
                data_type="user_logs",
                retention_period_days=730,  # 2 years for GDPR
                compliance_standard="gdpr",
                encryption_required=True,
                audit_required=True
            ),
            DataRetentionPolicy(
                data_type="audit_logs",
                retention_period_days=2555,  # 7 years
                compliance_standard="sox",
                encryption_required=True,
                audit_required=True
            ),
            DataRetentionPolicy(
                data_type="fraud_alerts",
                retention_period_days=1825,  # 5 years
                compliance_standard="pci_dss",
                encryption_required=True,
                audit_required=True
            )
        ]

        for policy in default_policies:
            self.add_policy(policy)

class ComplianceChecker:
    """Checks compliance with various regulatory standards."""

    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
        self.checks: List[ComplianceCheck] = []

    def run_compliance_check(self, standard: ComplianceStandard) -> ComplianceCheck:
        """Run a compliance check for a specific standard."""

        check_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()

        if standard == ComplianceStandard.PCI_DSS:
            result = self._check_pci_dss_compliance()
        elif standard == ComplianceStandard.GDPR:
            result = self._check_gdpr_compliance()
        elif standard == ComplianceStandard.SOX:
            result = self._check_sox_compliance()
        else:
            result = {
                "status": "not_implemented",
                "details": {"message": f"Compliance check for {standard.value} not implemented"}
            }

        check = ComplianceCheck(
            check_id=check_id,
            standard=standard.value,
            check_type="automated",
            status=result["status"],
            timestamp=timestamp,
            details=result["details"],
            remediation_required=result["status"] != "passed"
        )

        self.checks.append(check)
        logger.info(f"Compliance check completed: {standard.value} - {result['status']}")

        return check

    def _check_pci_dss_compliance(self) -> Dict[str, Any]:
        """Check PCI DSS compliance."""
        # Get recent audit entries
        recent_entries = self.audit_logger.get_entries(
            start_time=datetime.utcnow() - timedelta(days=30)
        )

        issues = []

        # Check for unencrypted data access
        unencrypted_access = [
            e for e in recent_entries
            if e.data_classification == "sensitive" and "encryption" not in e.details.get("security_measures", [])
        ]
        if unencrypted_access:
            issues.append(f"{len(unencrypted_access)} unencrypted sensitive data accesses")

        # Check for failed authentications
        failed_auths = [
            e for e in recent_entries
            if e.event_type == "authentication" and e.status == "failure"
        ]
        if len(failed_auths) > len(recent_entries) * 0.1:  # More than 10% failures
            issues.append("High authentication failure rate")

        return {
            "status": "passed" if not issues else "failed",
            "details": {
                "issues": issues,
                "total_events_checked": len(recent_entries),
                "unencrypted_accesses": len(unencrypted_access),
                "failed_authentications": len(failed_auths)
            }
        }

    def _check_gdpr_compliance(self) -> Dict[str, Any]:
        """Check GDPR compliance."""
        recent_entries = self.audit_logger.get_entries(
            start_time=datetime.utcnow() - timedelta(days=30)
        )

        issues = []

        # Check for data subject access requests handling
        dsar_entries = [
            e for e in recent_entries
            if "dsar" in e.details.get("request_type", "").lower()
        ]

        # Check response times (must be within 30 days)
        late_responses = [
            e for e in dsar_entries
            if e.details.get("response_days", 0) > 30
        ]
        if late_responses:
            issues.append(f"{len(late_responses)} late DSAR responses")

        return {
            "status": "passed" if not issues else "failed",
            "details": {
                "issues": issues,
                "dsar_requests": len(dsar_entries),
                "late_responses": len(late_responses)
            }
        }

    def _check_sox_compliance(self) -> Dict[str, Any]:
        """Check SOX compliance."""
        recent_entries = self.audit_logger.get_entries(
            event_type="configuration_change",
            start_time=datetime.utcnow() - timedelta(days=90)
        )

        issues = []

        # Check for unauthorized configuration changes
        unauthorized_changes = [
            e for e in recent_entries
            if e.status == "failure" or not e.details.get("authorized", False)
        ]
        if unauthorized_changes:
            issues.append(f"{len(unauthorized_changes)} unauthorized configuration changes")

        return {
            "status": "passed" if not issues else "failed",
            "details": {
                "issues": issues,
                "total_config_changes": len(recent_entries),
                "unauthorized_changes": len(unauthorized_changes)
            }
        }

    def get_compliance_status(self, standard: Optional[ComplianceStandard] = None) -> Dict[str, Any]:
        """Get overall compliance status."""
        if standard:
            recent_checks = [
                c for c in self.checks
                if c.standard == standard.value and c.timestamp > datetime.utcnow() - timedelta(days=30)
            ]
            if recent_checks:
                latest_check = max(recent_checks, key=lambda c: c.timestamp)
                return {
                    "standard": standard.value,
                    "status": latest_check.status,
                    "last_check": latest_check.timestamp.isoformat(),
                    "details": latest_check.details
                }

        # Overall status
        all_checks = [
            c for c in self.checks
            if c.timestamp > datetime.utcnow() - timedelta(days=30)
        ]

        standards_checked = set(c.standard for c in all_checks)
        passed_checks = len([c for c in all_checks if c.status == "passed"])
        total_checks = len(all_checks)

        return {
            "overall_status": "compliant" if passed_checks == total_checks else "non_compliant",
            "standards_checked": list(standards_checked),
            "passed_checks": passed_checks,
            "total_checks": total_checks,
            "compliance_rate": passed_checks / max(total_checks, 1)
        }

class PrivacyManager:
    """Manages data privacy and consent."""

    def __init__(self):
        self.consent_records: Dict[str, Dict[str, Any]] = {}
        self.data_processing_records: Dict[str, List[Dict[str, Any]]] = {}

    def record_consent(self, user_id: str, consent_type: str,
                      consented: bool, details: Dict[str, Any]):
        """Record user consent for data processing."""
        if user_id not in self.consent_records:
            self.consent_records[user_id] = {}

        self.consent_records[user_id][consent_type] = {
            "consented": consented,
            "timestamp": datetime.utcnow(),
            "details": details
        }

        logger.info(f"Consent recorded for user {user_id}: {consent_type} = {consented}")

    def check_consent(self, user_id: str, consent_type: str) -> bool:
        """Check if user has consented to specific data processing."""
        user_consents = self.consent_records.get(user_id, {})
        consent_record = user_consents.get(consent_type)

        if not consent_record:
            return False

        # Check if consent is still valid (not expired)
        if "expires_at" in consent_record.get("details", {}):
            expires_at = consent_record["details"]["expires_at"]
            if isinstance(expires_at, str):
                expires_at = datetime.fromisoformat(expires_at)
            if datetime.utcnow() > expires_at:
                return False

        return consent_record.get("consented", False)

    def record_data_processing(self, user_id: str, processing_type: str,
                             data_types: List[str], purpose: str):
        """Record data processing activity."""
        if user_id not in self.data_processing_records:
            self.data_processing_records[user_id] = []

        record = {
            "processing_type": processing_type,
            "data_types": data_types,
            "purpose": purpose,
            "timestamp": datetime.utcnow(),
            "consent_verified": self.check_consent(user_id, processing_type)
        }

        self.data_processing_records[user_id].append(record)

        if not record["consent_verified"]:
            logger.warning(f"Data processing without consent: {user_id} - {processing_type}")

    def get_privacy_report(self, user_id: str) -> Dict[str, Any]:
        """Generate privacy report for a user."""
        consents = self.consent_records.get(user_id, {})
        processing_records = self.data_processing_records.get(user_id, [])

        consent_summary = {}
        for consent_type, record in consents.items():
            consent_summary[consent_type] = {
                "consented": record["consented"],
                "timestamp": record["timestamp"].isoformat(),
                "valid": self.check_consent(user_id, consent_type)
            }

        processing_summary = {}
        for record in processing_records[-10:]:  # Last 10 records
            proc_type = record["processing_type"]
            if proc_type not in processing_summary:
                processing_summary[proc_type] = []
            processing_summary[proc_type].append({
                "timestamp": record["timestamp"].isoformat(),
                "consent_verified": record["consent_verified"],
                "data_types": record["data_types"]
            })

        return {
            "user_id": user_id,
            "consents": consent_summary,
            "recent_processing": processing_summary,
            "privacy_score": self._calculate_privacy_score(user_id)
        }

    def _calculate_privacy_score(self, user_id: str) -> float:
        """Calculate privacy compliance score for user."""
        consents = self.consent_records.get(user_id, {})
        processing_records = self.data_processing_records.get(user_id, [])

        if not consents:
            return 0.0

        total_consents = len(consents)
        valid_consents = sum(1 for c in consents.values() if c["consented"])

        consent_score = valid_consents / total_consents if total_consents > 0 else 0

        # Check processing without consent
        unconsented_processing = sum(
            1 for record in processing_records
            if not record["consent_verified"]
        )

        processing_penalty = min(unconsented_processing * 0.1, 0.5)

        return max(0, consent_score - processing_penalty)

# Global instances
_audit_logger = None
_compliance_checker = None
_privacy_manager = None
_data_retention_manager = None

def get_audit_logger() -> AuditLogger:
    """Get global audit logger."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger

def get_compliance_checker() -> ComplianceChecker:
    """Get global compliance checker."""
    global _compliance_checker
    if _compliance_checker is None:
        _compliance_checker = ComplianceChecker(get_audit_logger())
    return _compliance_checker

def get_privacy_manager() -> PrivacyManager:
    """Get global privacy manager."""
    global _privacy_manager
    if _privacy_manager is None:
        _privacy_manager = PrivacyManager()
    return _privacy_manager

def get_data_retention_manager() -> DataRetentionManager:
    """Get global data retention manager."""
    global _data_retention_manager
    if _data_retention_manager is None:
        _data_retention_manager = DataRetentionManager()
    return _data_retention_manager