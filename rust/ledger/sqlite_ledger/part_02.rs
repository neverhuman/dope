

impl CampaignLedger {
    pub fn initialize(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|error| io_error(parent, error))?;
        }
        let ledger = Self {
            path: path.to_path_buf(),
        };
        let connection = ledger.connection()?;
        connection.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=FULL;
             PRAGMA foreign_keys=ON;
             CREATE TABLE IF NOT EXISTS jobs (
                 job_id TEXT PRIMARY KEY,
                 spec_json BLOB NOT NULL,
                 state TEXT NOT NULL CHECK(state IN ('pending','leased','running','succeeded','failed','timed_out')),
                 attempts INTEGER NOT NULL DEFAULT 0,
                 lease_owner TEXT,
                 lease_until INTEGER,
                 heartbeat_at INTEGER,
                 last_error TEXT,
                 created_at INTEGER NOT NULL,
                 updated_at INTEGER NOT NULL
             );
             CREATE TABLE IF NOT EXISTS receipts (
                 receipt_id TEXT PRIMARY KEY,
                 job_id TEXT NOT NULL REFERENCES jobs(job_id),
                 attempt INTEGER NOT NULL,
                 receipt_json BLOB NOT NULL,
                 created_at INTEGER NOT NULL,
                 UNIQUE(job_id, attempt)
             );
             CREATE TABLE IF NOT EXISTS metadata (
                 key TEXT PRIMARY KEY,
                 value_json BLOB NOT NULL,
                 updated_at INTEGER NOT NULL
             );
             CREATE INDEX IF NOT EXISTS jobs_state_idx ON jobs(state, job_id);",
        )?;
        Ok(ledger)
    }

    pub fn open(path: &Path) -> Result<Self> {
        if !path.is_file() {
            return Err(DopeError::Data(format!(
                "campaign ledger does not exist: {}",
                path.display()
            )));
        }
        let ledger = Self {
            path: path.to_path_buf(),
        };
        let connection = ledger.connection()?;
        connection.pragma_update(None, "journal_mode", "WAL")?;
        connection.pragma_update(None, "foreign_keys", "ON")?;
        connection.execute_batch(
            "CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value_json BLOB NOT NULL,
                updated_at INTEGER NOT NULL
            );",
        )?;
        Ok(ledger)
    }

    fn connection(&self) -> Result<Connection> {
        let connection = Connection::open(&self.path)?;
        // Multiple short-lived handles are intentionally used by the scheduler
        // and by recovery tooling.  WAL removes reader/writer contention, but
        // writers can still briefly overlap; wait deterministically instead of
        // surfacing a transient SQLITE_BUSY failure.
        connection.busy_timeout(std::time::Duration::from_secs(5))?;
        Ok(connection)
    }

    pub fn insert_specs(&self, specs: &[JobSpec]) -> Result<usize> {
        let mut connection = self.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let timestamp = now();
        let mut inserted = 0;
        for spec in specs {
            let job_id = spec.id()?;
            let bytes = canonical_json(spec)?;
            inserted += transaction.execute(
                "INSERT OR IGNORE INTO jobs(job_id,spec_json,state,created_at,updated_at)
                 VALUES (?1,?2,'pending',?3,?3)",
                params![job_id, bytes, timestamp],
            )?;
            let existing: Vec<u8> = transaction.query_row(
                "SELECT spec_json FROM jobs WHERE job_id=?1",
                params![job_id],
                |row| row.get(0),
            )?;
            if existing != bytes {
                return Err(DopeError::Data(
                    "content-addressed job collision or mutated specification".into(),
                ));
            }
        }
        transaction.commit()?;
        Ok(inserted)
    }

    pub fn lease_next(&self, worker: &str, lease_seconds: u64) -> Result<Option<LeasedJob>> {
        self.lease_next_for(
            worker,
            lease_seconds,
            &["training_gold", "validation_select", "validation_cert"],
        )
    }

    pub fn lease_next_for(
        &self,
        worker: &str,
        lease_seconds: u64,
        allowed_phases: &[&str],
    ) -> Result<Option<LeasedJob>> {
        if worker.is_empty() || lease_seconds == 0 {
            return Err(DopeError::Data("invalid lease request".into()));
        }
        if allowed_phases.is_empty() {
            return Ok(None);
        }
        let mut connection = self.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let selected = {
            let mut statement = transaction.prepare(
                "SELECT job_id,spec_json,attempts FROM jobs
                 WHERE state='pending' ORDER BY job_id",
            )?;
            let rows = statement.query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, Vec<u8>>(1)?,
                    row.get::<_, u8>(2)?,
                ))
            })?;
            let mut selected = None;
            for row in rows {
                let row = row?;
                let spec: JobSpec = serde_json::from_slice(&row.1)?;
                if allowed_phases.contains(&spec.phase.as_str()) {
                    selected = Some(row);
                    break;
                }
            }
            selected
        };
        let Some((job_id, bytes, attempts)) = selected else {
            transaction.commit()?;
            return Ok(None);
        };
        let timestamp = now();
        let lease_until = timestamp.saturating_add(lease_seconds);
        let attempt = attempts.saturating_add(1);
        let changed = transaction.execute(
            "UPDATE jobs SET state='leased',attempts=?2,lease_owner=?3,lease_until=?4,
             heartbeat_at=?5,updated_at=?5 WHERE job_id=?1 AND state='pending'",
            params![job_id, attempt, worker, lease_until, timestamp],
        )?;
        if changed != 1 {
            return Err(DopeError::Data("job lease race was not serialized".into()));
        }
        transaction.commit()?;
        let spec: JobSpec = serde_json::from_slice(&bytes)?;
        spec.validate()?;
        Ok(Some(LeasedJob {
            job_id,
            attempt,
            lease_owner: worker.into(),
            lease_until_unix_seconds: lease_until,
            spec,
        }))
    }

    pub fn get_spec(&self, job_id: &str) -> Result<JobSpec> {
        let connection = self.connection()?;
        let bytes: Vec<u8> = connection.query_row(
            "SELECT spec_json FROM jobs WHERE job_id=?1",
            params![job_id],
            |row| row.get(0),
        )?;
        let spec: JobSpec = serde_json::from_slice(&bytes)?;
        if spec.id()? != job_id {
            return Err(DopeError::Data(
                "ledger job ID no longer matches stored bytes".into(),
            ));
        }
        Ok(spec)
    }

    pub fn mark_running(&self, job_id: &str, worker: &str) -> Result<()> {
        let connection = self.connection()?;
        let timestamp = now();
        let changed = connection.execute(
            "UPDATE jobs SET state='running',heartbeat_at=?3,updated_at=?3
             WHERE job_id=?1 AND state='leased' AND lease_owner=?2 AND lease_until>=?3",
            params![job_id, worker, timestamp],
        )?;
        if changed != 1 {
            return Err(DopeError::Data(
                "job cannot enter running state without a live matching lease".into(),
            ));
        }
        Ok(())
    }

    pub fn heartbeat(&self, job_id: &str, worker: &str, lease_seconds: u64) -> Result<()> {
        let connection = self.connection()?;
        let timestamp = now();
        let changed = connection.execute(
            "UPDATE jobs SET heartbeat_at=?3,lease_until=?4,updated_at=?3
             WHERE job_id=?1 AND lease_owner=?2 AND state IN ('leased','running')",
            params![
                job_id,
                worker,
                timestamp,
                timestamp.saturating_add(lease_seconds)
            ],
        )?;
        if changed != 1 {
            return Err(DopeError::Data("heartbeat does not own a live job".into()));
        }
        Ok(())
    }

    pub fn finish(&self, receipt: &JobReceipt) -> Result<()> {
        let spec = self.get_spec(&receipt.job_id)?;
        receipt.validate_against(&spec)?;
        let mut connection = self.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let (state, attempts, lease_owner): (String, u8, Option<String>) = transaction.query_row(
            "SELECT state,attempts,lease_owner FROM jobs WHERE job_id=?1",
            params![receipt.job_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )?;
        if !matches!(
            JobState::parse(&state)?,
            JobState::Leased | JobState::Running
        ) || attempts != receipt.attempt
            || lease_owner.as_deref() != Some(&receipt.worker)
        {
            return Err(DopeError::Data(
                "receipt is stale or job is already terminal".into(),
            ));
        }
        let receipt_id = receipt.id()?;
        let bytes = canonical_json(receipt)?;
        transaction.execute(
            "INSERT INTO receipts(receipt_id,job_id,attempt,receipt_json,created_at)
             VALUES (?1,?2,?3,?4,?5)",
            params![receipt_id, receipt.job_id, receipt.attempt, bytes, now()],
        )?;
        let next_state = match receipt.state {
            JobState::Succeeded => JobState::Succeeded,
            JobState::TimedOut => JobState::TimedOut,
            JobState::Failed => {
                if receipt.failure_class == Some(FailureClass::Infrastructure)
                    && receipt.attempt <= MAX_INFRASTRUCTURE_RETRIES
                {
                    JobState::Pending
                } else {
                    JobState::Failed
                }
            }
            _ => {
                return Err(DopeError::Data(
                    "worker receipt must be succeeded, failed, or timed_out".into(),
                ));
            }
        };
        transaction.execute(
            "UPDATE jobs SET state=?2,lease_owner=NULL,lease_until=NULL,heartbeat_at=NULL,
             last_error=?3,updated_at=?4 WHERE job_id=?1",
            params![receipt.job_id, next_state.as_str(), receipt.failure, now()],
        )?;
        transaction.commit()?;
        Ok(())
    }

    pub fn finish_signed(&self, receipt: &JobReceipt, key: &[u8; 32]) -> Result<()> {
        receipt.verify_signature(key)?;
        self.finish(receipt)
    }

    pub fn reclaim_expired(&self, timestamp: u64) -> Result<usize> {
        let mut connection = self.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        // SQLite INTEGER is signed; callers may use u64::MAX as a deterministic
        // "reclaim everything" sentinel during recovery/tests.
        let sqlite_timestamp = timestamp.min(i64::MAX as u64);
        let mut statement = transaction.prepare(
            "SELECT job_id,attempts FROM jobs
             WHERE state IN ('leased','running') AND lease_until<?1 ORDER BY job_id",
        )?;
        let expired: Vec<(String, u8)> = statement
            .query_map(params![sqlite_timestamp], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })?
            .collect::<std::result::Result<_, _>>()?;
        drop(statement);
        for (job_id, attempts) in &expired {
            let state = if *attempts <= MAX_INFRASTRUCTURE_RETRIES {
                JobState::Pending
            } else {
                JobState::TimedOut
            };
            transaction.execute(
                "UPDATE jobs SET state=?2,lease_owner=NULL,lease_until=NULL,heartbeat_at=NULL,
                 last_error='lease expired',updated_at=?3 WHERE job_id=?1",
                params![job_id, state.as_str(), sqlite_timestamp],
            )?;
        }
        transaction.commit()?;
        Ok(expired.len())
    }

    pub fn status(&self) -> Result<LedgerStatus> {
        let connection = self.connection()?;
        let mut counts = BTreeMap::new();
        for state in [
            JobState::Pending,
            JobState::Leased,
            JobState::Running,
            JobState::Succeeded,
            JobState::Failed,
            JobState::TimedOut,
        ] {
            let count: usize = connection.query_row(
                "SELECT COUNT(*) FROM jobs WHERE state=?1",
                params![state.as_str()],
                |row| row.get(0),
            )?;
            counts.insert(state, count);
        }
        let total: usize = counts.values().sum();
        let receipts = self.all_receipts()?;
        let completed_gold_labels = receipts
            .iter()
            .filter(|receipt| receipt.state == JobState::Succeeded && receipt.evidence.is_some())
            .count();
        let terminal =
            completed_gold_labels + counts[&JobState::Failed] + counts[&JobState::TimedOut];
        let measured: Option<serde_json::Value> = self.get_metadata("status_evidence")?;
        let router_regret = measured
            .as_ref()
            .and_then(|value| value.get("router_regret"))
            .and_then(serde_json::Value::as_f64);
        let ptf_v1 = measured
            .as_ref()
            .and_then(|value| value.get("ptf_v1"))
            .and_then(serde_json::Value::as_f64);
        let measured_coverage = measured
            .as_ref()
            .and_then(|value| value.get("coverage"))
            .and_then(serde_json::Value::as_f64);
        let production_score_available = measured
            .as_ref()
            .and_then(|value| value.get("production_score_available"))
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        Ok(LedgerStatus {
            format: "dope-campaign-ledger-status".into(),
            version: 1,
            total,
            training_progress: (total > 0).then_some(terminal as f64 / total as f64),
            completed_gold_labels,
            failed_jobs: counts[&JobState::Failed],
            timed_out_jobs: counts[&JobState::TimedOut],
            throughput_jobs_per_hour: throughput(&receipts),
            coverage: measured_coverage
                .or_else(|| (total > 0).then_some(terminal as f64 / total as f64)),
            // Router validation and PTF are imported only by the frozen evidence pipeline.
            // Job completion alone is not a quality measurement.
            current_router_validation_regret: router_regret,
            ptf_v1,
            production_score_available,
            counts,
            updated_unix_seconds: now(),
        })
    }

    pub fn successful_receipt(&self, job_id: &str) -> Result<Option<JobReceipt>> {
        let connection = self.connection()?;
        let bytes: Option<Vec<u8>> = connection
            .query_row(
                "SELECT r.receipt_json FROM receipts r JOIN jobs j ON j.job_id=r.job_id
                 WHERE r.job_id=?1 AND j.state='succeeded' ORDER BY r.attempt DESC LIMIT 1",
                params![job_id],
                |row| row.get(0),
            )
            .optional()?;
        bytes
            .map(|value| serde_json::from_slice(&value).map_err(Into::into))
            .transpose()
    }

    pub fn export_receipts(&self, path: &Path) -> Result<ContentHashes> {
        let receipts = self.all_receipts()?;
        let bytes = canonical_json(&receipts)?;
        fs::write(path, &bytes).map_err(|error| io_error(path, error))?;
        Ok(hashes(&bytes))
    }

    pub fn load_specs(path: &Path) -> Result<Vec<JobSpec>> {
        read_json(path)
    }

    pub fn all_specs(&self) -> Result<Vec<(String, JobState, JobSpec)>> {
        let connection = self.connection()?;
        let mut statement =
            connection.prepare("SELECT job_id,state,spec_json FROM jobs ORDER BY job_id")?;
        statement
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Vec<u8>>(2)?,
                ))
            })?
            .map(|row| {
                let (id, state, bytes) = row?;
                Ok((
                    id,
                    JobState::parse(&state)?,
                    serde_json::from_slice(&bytes)?,
                ))
            })
            .collect()
    }

    pub fn all_receipts(&self) -> Result<Vec<JobReceipt>> {
        let connection = self.connection()?;
        let mut statement =
            connection.prepare("SELECT receipt_json FROM receipts ORDER BY job_id,attempt")?;
        statement
            .query_map([], |row| row.get::<_, Vec<u8>>(0))?
            .map(|row| Ok(serde_json::from_slice(&row?)?))
            .collect()
    }

    pub fn put_metadata<T: Serialize>(&self, key: &str, value: &T) -> Result<()> {
        if key.is_empty() {
            return Err(DopeError::Data("metadata key cannot be empty".into()));
        }
        self.connection()?.execute(
            "INSERT INTO metadata(key,value_json,updated_at) VALUES (?1,?2,?3)
             ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json,
             updated_at=excluded.updated_at",
            params![key, canonical_json(value)?, now()],
        )?;
        Ok(())
    }

    pub fn get_metadata<T: for<'de> Deserialize<'de>>(&self, key: &str) -> Result<Option<T>> {
        let bytes: Option<Vec<u8>> = self
            .connection()?
            .query_row(
                "SELECT value_json FROM metadata WHERE key=?1",
                params![key],
                |row| row.get(0),
            )
            .optional()?;
        bytes
            .map(|bytes| serde_json::from_slice(&bytes).map_err(Into::into))
            .transpose()
    }
}