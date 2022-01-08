def test_import_config():
    from tai_chi_tuna.config import PhaseConfig
    assert PhaseConfig

def test_import_to_enrich():
    from tai_chi_tuna.flow.to_enrich import execute_enrich, set_enrich
    assert execute_enrich
    assert set_enrich

def test_import_to_quantify():
    from tai_chi_tuna.flow.to_quantify import (
        SIZE_DIMENSION,
        BATCH_SIZE,
        SEQUENCE_SIZE,
        IMAGE_SIZE,
        choose_xy,
        execute_quantify,
        save_qdict,
        load_qdict,
    )
    assert SIZE_DIMENSION
    assert BATCH_SIZE
    assert SEQUENCE_SIZE
    assert IMAGE_SIZE
    assert choose_xy
    assert execute_quantify
    assert save_qdict
    assert load_qdict

def test_import_to_model():
    from tai_chi_tuna.flow.to_model import (
        TaiChiCollate, TaiChiDataModule,
        choose_models,
        set_model,
        set_datamodule,
        assemble_model,
        set_opt_confs,
        AssembledModel,
        EntryDict,
        ParamWizard,
    )
    assert TaiChiCollate
    assert TaiChiDataModule
    assert choose_models
    assert set_model
    assert set_datamodule
    assert assemble_model
    assert set_opt_confs
    assert AssembledModel
    assert EntryDict
    assert ParamWizard

def test_import_to_train():
    from tai_chi_tuna.flow.to_train import (
        make_slug_name, 
        set_trainer,
        run_training,
    )
    assert make_slug_name
    assert set_trainer
    assert run_training

def test_import_trunk():
    from tai_chi_tuna.flow.trunk import (
        TaiChiStep,
        StepEnrich,
        StepQuantify,
        StepModeling, 
        StepTraining,
        TaiChiLearn,
    )
    assert TaiChiStep
    assert StepEnrich
    assert StepQuantify
    assert StepModeling
    assert StepTraining
    assert TaiChiLearn

