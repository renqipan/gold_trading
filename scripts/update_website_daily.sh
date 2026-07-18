#!/usr/bin/env bash

set -Eeuo pipefail
IFS=$'\n\t'
umask 077

export GIT_TERMINAL_PROMPT=0
export GIT_SSH_COMMAND="${GIT_SSH_COMMAND:-ssh -o BatchMode=yes}"

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly REMOTE="origin"
readonly BRANCH="main"
readonly PYTHON_BIN="${ROOT_DIR}/.venv-research/bin/python"
readonly -a DATA_FILES=(
  "public/data/gold_research_latest.json"
  "public/data/gold_price_series.json"
  "public/data/gold_backtest.json"
  "public/data/gold_forward_ledger.json"
)

die() {
  printf '错误：%s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "缺少命令：$1"
}

assert_no_git_operation() {
  local marker
  for marker in MERGE_HEAD CHERRY_PICK_HEAD REVERT_HEAD BISECT_LOG rebase-merge rebase-apply sequencer; do
    [[ ! -e "$(git rev-parse --git-path "${marker}")" ]] || die "Git 操作尚未结束：${marker}"
  done
}

assert_only_owned_changes() {
  local status_output status_line changed_path
  status_output="$(git status --porcelain --untracked-files=all)"
  while IFS= read -r status_line; do
    [[ -z "${status_line}" ]] && continue
    changed_path="${status_line:3}"
    case "${changed_path}" in
      public/data/gold_research_latest.json|public/data/gold_price_series.json|public/data/gold_backtest.json|public/data/gold_forward_ledger.json)
        ;;
      *)
        die "更新过程产生了非预期文件改动：${changed_path}"
        ;;
    esac
  done <<< "${status_output}"
}

read_as_of() {
  node -e "const d=require('./public/data/gold_research_latest.json'); process.stdout.write(d.asOf)"
}

read_is_final() {
  node -e "const d=require('./public/data/gold_research_latest.json'); process.stdout.write(String(d.priceStatus?.isFinal !== false))"
}

semantic_digest() {
  node <<'NODE'
const crypto = require("node:crypto");
const fs = require("node:fs");

function stable(value) {
  if (Array.isArray(value)) return value.map(stable);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.keys(value).sort().map((key) => [key, stable(value[key])]));
  }
  return value;
}

const paths = [
  "public/data/gold_research_latest.json",
  "public/data/gold_price_series.json",
  "public/data/gold_backtest.json",
  "public/data/gold_forward_ledger.json",
];
const payload = {};
for (const path of paths) {
  const value = JSON.parse(fs.readFileSync(path, "utf8"));
  if (path.endsWith("gold_research_latest.json")) {
    delete value.dataQuality;
    if (value.priceStatus) delete value.priceStatus.observedAt;
  }
  payload[path] = value;
}
process.stdout.write(
  crypto.createHash("sha256").update(JSON.stringify(stable(payload))).digest("hex"),
);
NODE
}

validate_history_append_only() {
  node - "${START_HEAD}" <<'NODE'
const { execFileSync } = require("node:child_process");
const fs = require("node:fs");
const startHead = process.argv[2];

function oldJson(path) {
  return JSON.parse(execFileSync("git", ["show", `${startHead}:${path}`], { encoding: "utf8" }));
}
function newJson(path) {
  return JSON.parse(fs.readFileSync(path, "utf8"));
}
function validateDated(path, revisionFields = []) {
  const before = oldJson(path);
  const after = newJson(path);
  const oldLatest = oldJson("public/data/gold_research_latest.json");
  const mutableTailDate = oldLatest.priceStatus?.isFinal === false ? oldLatest.asOf : null;
  const oldDates = before.map((row) => row.date);
  const newDates = after.map((row) => row.date);
  if (new Set(newDates).size !== newDates.length || [...newDates].sort().join() !== newDates.join()) {
    throw new Error(`日期重复或未排序：${path}`);
  }
  const afterByDate = new Map(after.map((row) => [row.date, row]));
  const newMin = newDates[0];
  for (const oldRow of before) {
    if (!afterByDate.has(oldRow.date)) {
      if (oldRow.date >= newMin) throw new Error(`历史中间日期消失：${path} ${oldRow.date}`);
      continue;
    }
    const newRow = afterByDate.get(oldRow.date);
    const stableOldRow = Object.fromEntries(
      Object.entries(oldRow).filter(([key]) => !revisionFields.includes(key)),
    );
    const stableNewRow = Object.fromEntries(
      Object.entries(newRow).filter(([key]) => !revisionFields.includes(key)),
    );
    if (JSON.stringify(stableNewRow) !== JSON.stringify(stableOldRow) && oldRow.date !== mutableTailDate) {
      throw new Error(`历史数据被改写：${path} ${oldRow.date}`);
    }
  }
  const oldMax = oldDates.at(-1);
  for (const row of after) {
    if (!oldDates.includes(row.date) && row.date <= oldMax) {
      throw new Error(`检测到历史回填而非末尾追加：${path} ${row.date}`);
    }
  }
}

validateDated("public/data/gold_price_series.json");
// FRED can publish a missing cash-yield observation after the initial run. It may
// revise strategy accounting, but never the gold benchmark, dates, or formal ledger.
validateDated("public/data/gold_backtest.json", ["equity", "drawdown", "position"]);

const ledgerPath = "public/data/gold_forward_ledger.json";
const beforeLedger = oldJson(ledgerPath);
const afterLedger = newJson(ledgerPath);
for (const key of ["strategyVersion", "executionEngineVersion", "configFingerprint", "start", "appendOnly"]) {
  if (beforeLedger[key] !== afterLedger[key]) throw new Error(`前瞻账本元数据变化：${key}`);
}
const priorRecords = beforeLedger.records || [];
const nextRecords = afterLedger.records || [];
if (JSON.stringify(nextRecords.slice(0, priorRecords.length)) !== JSON.stringify(priorRecords)) {
  throw new Error("前瞻账本不是 append-only");
}
NODE
}

cd "${ROOT_DIR}"
for command_name in git npm node; do
  require_command "${command_name}"
done
[[ -x "${PYTHON_BIN}" ]] || die "缺少研究环境，请先按 README 创建 .venv-research 并安装依赖"
node -e 'const [major,minor]=process.versions.node.split(".").map(Number); if(major<20||(major===20&&minor<9)) process.exit(1)' || die "Node.js 版本必须不低于 20.9"
"${PYTHON_BIN}" -c "import akshare, hmmlearn, numpy, pandas, requests, sklearn" || die "Python 研究依赖不完整"

assert_no_git_operation
[[ "$(git branch --show-current)" == "${BRANCH}" ]] || die "请先切换到 main 分支"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || die "工作树不干净；请先提交、暂存或移走现有改动"

git_common_dir="$(git rev-parse --git-common-dir)"
if [[ "${git_common_dir}" != /* ]]; then
  git_common_dir="${ROOT_DIR}/${git_common_dir}"
fi
readonly LOCK_DIR="${git_common_dir%/}/gold-daily-update.lock"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  die "已有更新任务在运行；若确认没有进程，请删除 ${LOCK_DIR}"
fi
printf 'pid=%s started=%s\n' "$$" "$(date -u +%FT%TZ)" >"${LOCK_DIR}/owner"
readonly STEP_LOG="${LOCK_DIR}/step.log"

run_quiet_step() {
  local label="$1"
  shift
  : >"${STEP_LOG}"
  printf '%s ... ' "${label}"
  if "$@" >"${STEP_LOG}" 2>&1; then
    printf '完成\n'
    return 0
  fi
  printf '失败\n' >&2
  tail -n 80 "${STEP_LOG}" >&2 || true
  die "${label}失败"
}

START_HEAD=""
COMMIT_CREATED=0
restore_owned() {
  [[ -n "${START_HEAD}" ]] || return 0
  [[ "$(git rev-parse HEAD 2>/dev/null || true)" == "${START_HEAD}" ]] || return 0
  git restore --source="${START_HEAD}" --staged --worktree -- "${DATA_FILES[@]}" >/dev/null 2>&1 || true
}
cleanup() {
  local rc=$?
  trap - EXIT
  if [[ "${rc}" -ne 0 && "${COMMIT_CREATED}" -eq 0 ]]; then
    restore_owned
  fi
  rm -f "${LOCK_DIR}/owner" 2>/dev/null || true
  rm -f "${STEP_LOG}" 2>/dev/null || true
  rmdir "${LOCK_DIR}" 2>/dev/null || true
  exit "${rc}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

run_quiet_step '[1/4] 同步 GitHub main' git fetch --quiet --prune "${REMOTE}" "${BRANCH}"
local_head="$(git rev-parse HEAD)"
remote_head="$(git rev-parse "${REMOTE}/${BRANCH}")"
if [[ "${local_head}" != "${remote_head}" ]]; then
  if git merge-base --is-ancestor "${local_head}" "${remote_head}"; then
    git merge --quiet --ff-only "${remote_head}"
  elif git merge-base --is-ancestor "${remote_head}" "${local_head}"; then
    die "本地 main 有尚未推送的提交，请先执行 git push origin main"
  else
    die "本地 main 与 origin/main 已分叉，请人工处理"
  fi
fi

START_HEAD="$(git rev-parse HEAD)"
readonly BASE_REMOTE_HEAD="$(git rev-parse "${REMOTE}/${BRANCH}")"
readonly OLD_AS_OF="$(read_as_of)"
readonly OLD_DIGEST="$(semantic_digest)"

run_quiet_step '[2/4] 更新行情与模型' "${PYTHON_BIN}" research/gold_research_pipeline.py
assert_only_owned_changes
run_quiet_step '[3/4] 验证策略与网站' npm run --silent verify
assert_only_owned_changes

NEW_AS_OF="$(read_as_of)"
[[ "${NEW_AS_OF}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || die "网站数据缺少有效 asOf 日期"
if [[ "${NEW_AS_OF}" > "$(TZ=Asia/Shanghai date +%F)" ]]; then
  die "网站数据日期不能晚于北京时间当天：${NEW_AS_OF}"
fi
[[ "${NEW_AS_OF}" > "${OLD_AS_OF}" || "${NEW_AS_OF}" == "${OLD_AS_OF}" ]] || die "网站数据日期倒退：${OLD_AS_OF} -> ${NEW_AS_OF}"

NEW_IS_FINAL="$(read_is_final)"
NEW_DIGEST="$(semantic_digest)"
if [[ "${NEW_AS_OF}" == "${OLD_AS_OF}" ]]; then
  if [[ "${NEW_DIGEST}" == "${OLD_DIGEST}" ]]; then
    if [[ "${FORCE_PUBLISH_TIMESTAMP:-0}" != "1" ]]; then
      restore_owned
      printf '[4/4] 发布网站 ... 无变化，无需提交\n'
      exit 0
    fi
  fi
fi

validate_history_append_only
git add -- "${DATA_FILES[@]}"
git diff --cached --quiet && die "asOf 已更新但没有可提交的数据变化"
git diff --cached --check

printf '[4/4] 发布网站 ... '
git fetch --quiet "${REMOTE}" "${BRANCH}"
[[ "$(git rev-parse "${REMOTE}/${BRANCH}")" == "${BASE_REMOTE_HEAD}" ]] || die "origin/main 在更新期间发生变化；禁止自动 rebase 或 force push"

if [[ "${NEW_DIGEST}" == "${OLD_DIGEST}" ]]; then
  commit_subject="Refresh gold website at $(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M') Asia/Shanghai"
elif [[ "${NEW_IS_FINAL}" == "true" ]]; then
  commit_subject="Update gold data through ${NEW_AS_OF} close"
else
  commit_subject="Update gold intraday snapshot for ${NEW_AS_OF}"
fi
git commit --quiet -m "${commit_subject}" -m "Gold-Data-Automation: v2"
COMMIT_CREATED=1
git push --quiet "${REMOTE}" HEAD:refs/heads/main
git fetch --quiet "${REMOTE}" "${BRANCH}"
[[ "$(git rev-parse HEAD)" == "$(git rev-parse "${REMOTE}/${BRANCH}")" ]] || die "推送后本地与远端 main 不一致"
printf '完成（%s）\n' "${NEW_AS_OF}"
