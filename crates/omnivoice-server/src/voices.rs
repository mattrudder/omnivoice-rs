use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use omnivoice_infer::{
    contracts::{ReferenceAudioInput, VoiceClonePrompt},
    pipeline::Phase3Pipeline,
};
use serde::Deserialize;

use crate::error::ServerError;

#[derive(Clone, Debug)]
pub enum PreloadedVoice {
    Clone(VoiceClonePrompt),
    Design(String),
}

impl PreloadedVoice {
    /// Discriminant for the listing endpoint. A clone carries tokenized
    /// reference audio; a design carries an instruct string, which the request
    /// path will overwrite an explicit `instruct` with.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Clone(_) => "clone",
            Self::Design(_) => "design",
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct VoiceConfig {
    #[serde(default)]
    pub default_voice: Option<String>,
    #[serde(default)]
    pub voices: HashMap<String, VoiceEntry>,
}

#[derive(Debug, Deserialize)]
pub struct VoiceEntry {
    #[serde(default)]
    pub ref_audio: Option<PathBuf>,
    #[serde(default)]
    pub ref_text: Option<String>,
    #[serde(default)]
    pub instruct: Option<String>,
}

impl VoiceConfig {
    pub fn from_path(path: &Path) -> Result<Self, ServerError> {
        let content = fs::read_to_string(path).map_err(|e| {
            ServerError::internal(format!(
                "failed to read voices config {}: {e}",
                path.display()
            ))
        })?;
        let mut config: Self = toml::from_str(&content).map_err(|e| {
            ServerError::internal(format!(
                "failed to parse voices config {}: {e}",
                path.display()
            ))
        })?;

        // Resolve ref_audio paths relative to the config file's directory.
        let base = path.parent().unwrap_or(Path::new("."));
        for entry in config.voices.values_mut() {
            if let Some(ref mut ref_audio) = entry.ref_audio {
                if ref_audio.is_relative() {
                    *ref_audio = base.join(&*ref_audio);
                }
            }
        }

        Ok(config)
    }

    pub fn validate(&self) -> Result<(), ServerError> {
        if let Some(ref name) = self.default_voice {
            if !self.voices.contains_key(name) {
                return Err(ServerError::internal(format!(
                    "voices config: default_voice '{name}' is not defined in [voices]"
                )));
            }
        }
        for (name, entry) in &self.voices {
            match (&entry.ref_audio, &entry.instruct) {
                (None, None) => {
                    return Err(ServerError::internal(format!(
                        "voices config: voice '{name}' must specify either ref_audio or instruct"
                    )));
                }
                (Some(_), Some(_)) => {
                    return Err(ServerError::internal(format!(
                        "voices config: voice '{name}' cannot specify both ref_audio and instruct"
                    )));
                }
                (Some(ref_audio), None) => {
                    if !ref_audio.exists() {
                        return Err(ServerError::internal(format!(
                            "voices config: ref_audio for voice '{name}' not found: {}",
                            ref_audio.display()
                        )));
                    }
                }
                (None, Some(instruct)) => {
                    if instruct.trim().is_empty() {
                        return Err(ServerError::internal(format!(
                            "voices config: instruct for voice '{name}' must not be empty"
                        )));
                    }
                }
            }
        }
        Ok(())
    }
}

pub fn preload_voices(
    config: &VoiceConfig,
    pipeline: &Phase3Pipeline,
) -> Result<HashMap<String, PreloadedVoice>, ServerError> {
    let mut voices = HashMap::new();
    for (name, entry) in &config.voices {
        let voice = match (&entry.ref_audio, &entry.instruct) {
            (Some(ref_audio), _) => {
                let input = ReferenceAudioInput::from_path(ref_audio);
                let prompt = pipeline
                    .create_voice_clone_prompt_from_audio(
                        &input,
                        entry.ref_text.as_deref(),
                        true,
                        None,
                    )
                    .map_err(|e| {
                        ServerError::internal(format!("failed to tokenize voice '{name}': {e}"))
                    })?;
                PreloadedVoice::Clone(prompt)
            }
            (None, Some(instruct)) => PreloadedVoice::Design(instruct.clone()),
            (None, None) => unreachable!("validate() ensures one is set"),
        };
        voices.insert(name.clone(), voice);
    }
    Ok(voices)
}
