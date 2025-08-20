trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()