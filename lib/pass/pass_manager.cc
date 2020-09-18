//===- pass_manager.cc ----------------------------------------------------===//
//
// Copyright (C) 2019-2020 Alibaba Group Holding Limited.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "halo/lib/pass/pass_manager.h"

namespace halo {

class PassManagerImpl {
 public:
  explicit PassManagerImpl(GlobalContext* ctx) : ctx_(*ctx) {}

  Pass* Add(std::unique_ptr<ModulePass> pass);
  Pass* Add(std::unique_ptr<FunctionPass> pass);
  Pass* Add(std::unique_ptr<BasicBlockPass> pass);

  GlobalContext& GetGobalContext() { return ctx_; }

  Status Run(Module* module);

  void Print(std::ostream& os) const;

 private:
  FunctionPassManager* GetFunctionPassManager();

  GlobalContext& ctx_;
  std::list<std::unique_ptr<ModulePass>> passes_;
}; // namespace halo

PassManager::PassManager(GlobalContext& ctx)
    : impl_(std::make_unique<PassManagerImpl>(&ctx)) {}

PassManager::~PassManager() {}

Pass* PassManager::Add(std::unique_ptr<ModulePass> pass) {
  return impl_->Add(std::move(pass));
}

Pass* PassManager::Add(std::unique_ptr<FunctionPass> pass) {
  return impl_->Add(std::move(pass));
}

Pass* PassManager::Add(std::unique_ptr<BasicBlockPass> pass) {
  return impl_->Add(std::move(pass));
}

Status PassManager::Run(Module* module) { return impl_->Run(module); }

void PassManager::Print(std::ostream& os) const { impl_->Print(os); }

void PassManager::Dump() const { Print(GlobalContext::Dbgs()); }

// BasicBlockPassManager is a function level pass that contains basic block
// passes.
class BasicBlockPassManager final : public FunctionPass {
 public:
  BasicBlockPassManager() : FunctionPass("BasicBlockPassManager") {}
  bool RunOnFunction(Function* function) override {
    bool changed = true;
    while (changed) {
      changed = false;
      for (auto& bb : *function) {
        for (auto& fp : passes_) {
          changed |= fp->RunOnBasicBlock(bb.get());
        }
      }
      if (!changed) {
        break;
      }
    }
    return changed;
  }
  void AddPass(std::unique_ptr<BasicBlockPass> pass) {
    passes_.push_back(std::move(pass));
  }

  void Print(std::ostream& os) const override {
    os << Name() << "\n";
    for (auto& pass : passes_) {
      pass->Print(os);
    }
  }

  bool IsPassManager() const noexcept override { return true; }

 private:
  std::list<std::unique_ptr<BasicBlockPass>> passes_;
};

// FunctionPassManager is a module level pass that contains function passes.
class FunctionPassManager final : public ModulePass {
 public:
  FunctionPassManager() : ModulePass("FunctionPassManager") {}
  bool RunOnModule(Module* module) override {
    bool changed = true;
    while (changed) {
      changed = false;
      for (auto& func : *module) {
        for (auto& fp : passes_) {
          changed |= fp->RunOnFunction(func.get());
        }
      }
    }
    return changed;
  }

  void AddPass(std::unique_ptr<FunctionPass> pass) {
    passes_.push_back(std::move(pass));
  }

  BasicBlockPassManager* GetBasicBlockPassManager() {
    if (passes_.empty() || !passes_.back()->IsPassManager()) {
      passes_.push_back(std::make_unique<BasicBlockPassManager>());
    }
    BasicBlockPassManager* fpm =
        Downcast<BasicBlockPassManager>(passes_.back().get());
    return fpm;
  }

  void Print(std::ostream& os) const override {
    os << Name() << "\n";
    for (auto& pass : passes_) {
      pass->Print(os);
    }
  }

  bool IsPassManager() const noexcept override { return true; }

 private:
  std::list<std::unique_ptr<FunctionPass>> passes_;
};

Pass* PassManagerImpl::Add(std::unique_ptr<ModulePass> pass) {
  passes_.push_back(std::move(pass));
  return passes_.back().get();
}

Pass* PassManagerImpl::Add(std::unique_ptr<FunctionPass> pass) {
  Pass* ret = pass.get();
  FunctionPassManager* fpm = GetFunctionPassManager();
  fpm->AddPass(std::move(pass));
  return ret;
}

Pass* PassManagerImpl::Add(std::unique_ptr<BasicBlockPass> pass) {
  Pass* ret = pass.get();
  FunctionPassManager* fpm = GetFunctionPassManager();
  BasicBlockPassManager* bpm = fpm->GetBasicBlockPassManager();
  bpm->AddPass(std::move(pass));
  return ret;
}

Status PassManagerImpl::Run(Module* module) {
  for (auto& pass : passes_) {
    pass->RunOnModule(module);
  }
  return Status::SUCCESS;
}

FunctionPassManager* PassManagerImpl::GetFunctionPassManager() {
  if (passes_.empty() || !passes_.back()->IsPassManager()) {
    passes_.push_back(std::make_unique<FunctionPassManager>());
  }
  FunctionPassManager* fpm =
      Downcast<FunctionPassManager>(passes_.back().get());
  return fpm;
}

void PassManagerImpl::Print(std::ostream& os) const {
  for (auto& pass : passes_) {
    pass->Print(os);
  }
}

} // end namespace halo