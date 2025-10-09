use crate::config::{LogFormat, LogOutput, LoggingConfig, RotationPolicy};
use opentelemetry_appender_tracing::layer::OpenTelemetryTracingBridge;
use opentelemetry_sdk::logs::SdkLoggerProvider;
use std::fs;
use tracing_appender::non_blocking::{NonBlocking, WorkerGuard};
use tracing_subscriber::fmt;
use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

pub fn init_logging(config: &LoggingConfig) -> SdkLoggerProvider {
    // Simple stdout exporter for now
    let exporter = opentelemetry_stdout::LogExporter::default();

    let logger_provider = SdkLoggerProvider::builder()
        .with_simple_exporter(exporter)
        .build();

    let telemetry_layer = OpenTelemetryTracingBridge::new(&logger_provider);

    // Build the subscriber based on config
    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&config.level));

    let subscriber = tracing_subscriber::registry()
        .with(env_filter)
        .with(telemetry_layer);

    // Apply format and writer in one go
    match config.output {
        LogOutput::Stdout => match config.format {
            LogFormat::Pretty => subscriber.with(fmt::layer().pretty()).init(),
            LogFormat::Json => subscriber.with(fmt::layer().json()).init(),
            LogFormat::Compact => subscriber.with(fmt::layer().compact()).init(),
        },
        LogOutput::File => {
            if let Some(file_config) = &config.file {
                let (writer, _guard) = create_file_writer(file_config);
                // Leak the guard to keep it alive for the program duration
                Box::leak(Box::new(_guard));

                match config.format {
                    LogFormat::Pretty => subscriber
                        .with(fmt::layer().pretty().with_writer(writer))
                        .init(),
                    LogFormat::Json => subscriber
                        .with(fmt::layer().json().with_writer(writer))
                        .init(),
                    LogFormat::Compact => subscriber
                        .with(fmt::layer().compact().with_writer(writer))
                        .init(),
                }
            } else {
                eprintln!("File output requested but no file config provided");
                subscriber.with(fmt::layer()).init();
            }
        }
        LogOutput::Both => {
            // For simplicity, just use stdout for now
            // Properly implementing "both" requires more complex layering
            tracing::warn!("'Both' output not fully implemented, using stdout only");
            match config.format {
                LogFormat::Pretty => subscriber.with(fmt::layer().pretty()).init(),
                LogFormat::Json => subscriber.with(fmt::layer().json()).init(),
                LogFormat::Compact => subscriber.with(fmt::layer().compact()).init(),
            }
        }
    }

    tracing::info!(
        "Logging initialized: level={}, format={:?}",
        config.level,
        config.format
    );
    logger_provider
}

fn create_file_writer(
    file_config: &crate::config::FileLoggingConfig,
) -> (NonBlocking, WorkerGuard) {
    // Create directory
    if let Err(e) = fs::create_dir_all(&file_config.directory) {
        eprintln!("Failed to create log directory: {e}. Using stdout.");
        return tracing_appender::non_blocking(std::io::stdout());
    }

    // Create rolling file appender
    let appender = match file_config.rotation_policy {
        RotationPolicy::Daily => {
            tracing_appender::rolling::daily(&file_config.directory, &file_config.prefix)
        }
        RotationPolicy::Hourly => {
            tracing_appender::rolling::hourly(&file_config.directory, &file_config.prefix)
        }
        RotationPolicy::Size => {
            tracing::warn!("Size-based rotation not supported, using daily");
            tracing_appender::rolling::daily(&file_config.directory, &file_config.prefix)
        }
    };

    tracing_appender::non_blocking(appender)
}

pub fn shutdown_logging(logger_provider: SdkLoggerProvider) {
    if let Err(err) = logger_provider.shutdown() {
        eprintln!("Failed to shutdown logger provider: {err}");
    }
}

// TODO: init_metrics
// TODO: init_tracing
// TODO: top level init for all telemetry
// TODO: top level shutdown for all telemetry
