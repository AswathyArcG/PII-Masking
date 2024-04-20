
from typing import List, Dict, Optional, Tuple, Type
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_anonymizer.entities import (
    OperatorConfig,
)
from presidio_analyzer.nlp_engine import (
    NlpEngine,
    NlpEngineProvider,
)
from presidio_analyzer.nlp_engine import TransformersNlpEngine, NerModelConfiguration


class PiiMaskingService():

    def analyze(self, text: str, model: str):

        entitiesToRecognize=['UK_NHS','EMAIL','AU_ABN','CRYPTO','ID','URL',
                             'AU_MEDICARE','IN_PAN','ORGANIZATION','IN_AADHAAR',
                             'SG_NRIC_FIN','EMAIL_ADDRESS','AU_ACN','US_DRIVER_LICENSE',
                             'IP_ADDRESS','DATE_TIME','LOCATION','PERSON','CREDIT_CARD',
                             'IBAN_CODE','US_BANK_NUMBER','PHONE_NUMBER','MEDICAL_LICENSE',
                             'US_SSN','AU_TFN','US_PASSPORT','US_ITIN','NRP','AGE','GENERIC_PII'
                             ]

        if model == "HuggingFace/obi/deid_roberta_i2b2":
            nlp_engine, registry= self.create_nlp_engine_with_transformers("obi/deid_roberta_i2b2")
        elif model == "flair/ner-english-large":
            nlp_engine, registry= self.create_nlp_engine_with_flair("flair/ner-english-large")

        analyzer = AnalyzerEngine(nlp_engine=nlp_engine, registry=registry)

        results = analyzer.analyze(text=text, entities=entitiesToRecognize, language='en') 
        print("analyzer results:")
        print(results)

        return results
    
    
    def anonymize(
            self,
            text: str,
            operator: str,
            model: str
            # analyze_results: List[RecognizerResult],
        ):
        operator_config = None
        encrypt_key = "WmZq4t7w!z%C&F)J"

        if operator == 'mask':
            operator_config = {
                "type": "mask",
                "masking_char": "*",
                "chars_to_mask": 15,
                "from_end": False,
            }
        elif operator == "encrypt":
            operator_config = {"key": encrypt_key}
        elif operator == "highlight":
            operator_config = {"lambda": lambda x: x}


        if operator == "highlight":
            operator = "custom"


        analyzer_result = self.analyze(text, model)

        engine = AnonymizerEngine()

            # Invoke the anonymize function with the text, analyzer results and
            # Operators to define the anonymization type.
        result = engine.anonymize(
            text=text,
            operators={"DEFAULT": OperatorConfig(operator, operator_config)},
            analyzer_results=analyzer_result
        )
        print("res:")
        print(result)
        print(result.text)
        print(type(result.text))


        return result.text
    
    
    def create_nlp_engine_with_flair(
            self,
            model_path: str,
    ) -> Tuple[NlpEngine, RecognizerRegistry]:
        """
        Instantiate an NlpEngine with a FlairRecognizer and a small spaCy model.
        The FlairRecognizer would return results from Flair models, the spaCy model
        would return NlpArtifacts such as POS and lemmas.
        :param model_path: Flair model path.
        """
        from flair_recognizer import FlairRecognizer

        registry = RecognizerRegistry()
        registry.load_predefined_recognizers()

        # there is no official Flair NlpEngine, hence we load it as an additional recognizer

        # if not spacy.util.is_package("en_core_web_sm"):
        #     spacy.cli.download("en_core_web_sm")
        # Using a small spaCy model + a Flair NER model
        flair_recognizer = FlairRecognizer(model_path=model_path)
        nlp_configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }
        registry.add_recognizer(flair_recognizer)
        registry.remove_recognizer("SpacyRecognizer")

        nlp_engine = NlpEngineProvider(nlp_configuration=nlp_configuration).create_engine()

        return nlp_engine, registry
    

    def create_nlp_engine_with_transformers(
            self,
            model_path: str,
    ) -> Tuple[NlpEngine, RecognizerRegistry]:
        """
        Instantiate an NlpEngine with a TransformersRecognizer and a small spaCy model.
        The TransformersRecognizer would return results from Transformers models, the spaCy model
        would return NlpArtifacts such as POS and lemmas.
        :param model_path: HuggingFace model path.
        """
        print(f"Loading Transformers model: {model_path} of type {type(model_path)}")

        nlp_configuration = {
            "nlp_engine_name": "transformers",
            "models": [
                {
                    "lang_code": "en",
                    "model_name": {"spacy": "en_core_web_sm", "transformers": model_path},
                }
            ],
            "ner_model_configuration": {
                "model_to_presidio_entity_mapping": {
                    "PER": "PERSON",
                    "PERSON": "PERSON",
                    "LOC": "LOCATION",
                    "LOCATION": "LOCATION",
                    "GPE": "LOCATION",
                    "ORG": "ORGANIZATION",
                    "ORGANIZATION": "ORGANIZATION",
                    "NORP": "NRP",
                    "AGE": "AGE",
                    "ID": "ID",
                    "EMAIL": "EMAIL",
                    "PATIENT": "PERSON",
                    "STAFF": "PERSON",
                    "HOSP": "ORGANIZATION",
                    "PATORG": "ORGANIZATION",
                    "DATE": "DATE_TIME",
                    "TIME": "DATE_TIME",
                    "PHONE": "PHONE_NUMBER",
                    "HCW": "PERSON",
                    "HOSPITAL": "ORGANIZATION",
                    "FACILITY": "LOCATION",
                },
                "low_confidence_score_multiplier": 0.4,
                "low_score_entity_names": ["ID"],
                "labels_to_ignore": [
                    "CARDINAL",
                    "EVENT",
                    "LANGUAGE",
                    "LAW",
                    "MONEY",
                    "ORDINAL",
                    "PERCENT",
                    "PRODUCT",
                    "QUANTITY",
                    "WORK_OF_ART",
                ],
            },
        }

        nlp_engine = NlpEngineProvider(nlp_configuration=nlp_configuration).create_engine()

        registry = RecognizerRegistry()
        registry.load_predefined_recognizers(nlp_engine=nlp_engine)

        return nlp_engine, registry
