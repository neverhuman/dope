fn run_campaign(command: CampaignCommand) -> Result<()> {
    match command {
        command @ (CampaignCommand::FreezeDeepImplementation { .. } | CampaignCommand::CreateReceiptKey { .. } | CampaignCommand::Freeze { .. } | CampaignCommand::Serve { .. } | CampaignCommand::Worker { .. } | CampaignCommand::VerifyCorpus { .. } | CampaignCommand::ExportCorpusSample { .. } | CampaignCommand::PlanCohort { .. } | CampaignCommand::PlanDeepCohorts { .. } | CampaignCommand::MaterializeDeepCohort { .. } | CampaignCommand::BuildDeepEvidence { .. } | CampaignCommand::SelectDeepDiscovery { .. } | CampaignCommand::QualifyDeepConfirmation { .. } | CampaignCommand::SelectDeepFinal { .. } | CampaignCommand::BuildDeepArtifactManifest { .. } | CampaignCommand::BuildDeepModelCard { .. } | CampaignCommand::ReportDeepCampaign { .. } | CampaignCommand::PlanValidationCert { .. } | CampaignCommand::RunBaseline { .. } | CampaignCommand::SummarizeEvidence { .. } | CampaignCommand::RunGoldMatrix { .. }) => run_campaign_group_01(command),
        command @ (CampaignCommand::RunRoutedCert { .. } | CampaignCommand::SignGoldBlocks { .. } | CampaignCommand::TrainRouter { .. } | CampaignCommand::BenchmarkRouter { .. } | CampaignCommand::EvaluateGoldCell { .. } | CampaignCommand::Status { .. } | CampaignCommand::ExportMetrics { .. } | CampaignCommand::ExportValidationCertMetrics { .. } | CampaignCommand::FreezeRouter { .. } | CampaignCommand::OpenValidationCert { .. }) => run_campaign_group_02(command),
        command @ CampaignCommand::BuildRc { .. } => run_campaign_group_03(command),
    }
}

include!("campaign_commands/group_01.rs");
include!("campaign_commands/group_02.rs");
include!("campaign_commands/group_03.rs");
