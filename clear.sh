#!/bin/bash
# ============================================================
# clean-claude-proxy.sh
# 清理 Claude Code / Codex 的代理配置（如 CCSwitch 残留）
# 用法：chmod +x clean-claude-proxy.sh && ./clean-claude-proxy.sh
# ============================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo ""
echo "🧹 Claude Code 代理配置清理工具"
echo "================================"
echo ""

CHANGED=0

# ---- 1. ~/.claude/settings.json ----
SETTINGS_FILE="$HOME/.claude/settings.json"
if [ -f "$SETTINGS_FILE" ]; then
    if grep -q "PROXY_MANAGED\|ANTHROPIC_BASE_URL\|ANTHROPIC_AUTH_TOKEN" "$SETTINGS_FILE" 2>/dev/null; then
        echo -e "${YELLOW}[发现]${NC} $SETTINGS_FILE 包含代理配置"
        echo "  清理前内容："
        cat "$SETTINGS_FILE" | sed 's/^/    /'
        
        # 用 python3 安全地修改 JSON（保留其他配置）
        python3 -c "
import json, sys
with open('$SETTINGS_FILE', 'r') as f:
    data = json.load(f)
env = data.get('env', {})
for key in ['ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_BASE_URL', 'ANTHROPIC_API_KEY']:
    env.pop(key, None)
data['env'] = env
with open('$SETTINGS_FILE', 'w') as f:
    json.dump(data, f, indent=2)
    f.write('\n')
"
        echo -e "  ${GREEN}✅ 已清理 env 中的代理变量${NC}"
        echo "  清理后内容："
        cat "$SETTINGS_FILE" | sed 's/^/    /'
        CHANGED=$((CHANGED + 1))
    else
        echo -e "${GREEN}[OK]${NC} $SETTINGS_FILE 无代理配置"
    fi
else
    echo -e "${GREEN}[OK]${NC} $SETTINGS_FILE 不存在，跳过"
fi

echo ""

# ---- 2. ~/.codex/auth.json ----
CODEX_AUTH="$HOME/.codex/auth.json"
if [ -f "$CODEX_AUTH" ]; then
    if grep -q "PROXY_MANAGED" "$CODEX_AUTH" 2>/dev/null; then
        echo -e "${YELLOW}[发现]${NC} $CODEX_AUTH 包含代理配置"
        echo "  清理前内容："
        cat "$CODEX_AUTH" | sed 's/^/    /'
        
        echo '{}' > "$CODEX_AUTH"
        
        echo -e "  ${GREEN}✅ 已重置为空配置${NC}"
        CHANGED=$((CHANGED + 1))
    else
        echo -e "${GREEN}[OK]${NC} $CODEX_AUTH 无代理配置"
    fi
else
    echo -e "${GREEN}[OK]${NC} $CODEX_AUTH 不存在，跳过"
fi

echo ""

# ---- 3. ~/.claude.json（清除旧账号 userID） ----
CLAUDE_JSON="$HOME/.claude.json"
if [ -f "$CLAUDE_JSON" ]; then
    if grep -q '"userID"' "$CLAUDE_JSON" 2>/dev/null; then
        echo -e "${YELLOW}[发现]${NC} $CLAUDE_JSON 包含旧账号 userID"
        
        python3 -c "
import json
with open('$CLAUDE_JSON', 'r') as f:
    data = json.load(f)
removed = []
for key in ['userID', 'hasAvailableSubscription', 'subscriptionNoticeCount', 'clientDataCache']:
    if key in data:
        del data[key]
        removed.append(key)
data['numStartups'] = 0
data['hasCompletedOnboarding'] = False
with open('$CLAUDE_JSON', 'w') as f:
    json.dump(data, f, indent=2)
    f.write('\n')
print('  移除字段: ' + ', '.join(removed))
"
        echo -e "  ${GREEN}✅ 已清除旧账号信息${NC}"
        CHANGED=$((CHANGED + 1))
    else
        echo -e "${GREEN}[OK]${NC} $CLAUDE_JSON 无旧账号 userID"
    fi
else
    echo -e "${GREEN}[OK]${NC} $CLAUDE_JSON 不存在，跳过"
fi

echo ""

# ---- 4. 检查 shell 环境变量 ----
echo "检查 shell 配置文件..."
SHELL_FILES=("$HOME/.zshrc" "$HOME/.zprofile" "$HOME/.bash_profile" "$HOME/.bashrc")
SHELL_CLEAN=true
for f in "${SHELL_FILES[@]}"; do
    if [ -f "$f" ] && grep -qE "ANTHROPIC_AUTH_TOKEN|ANTHROPIC_BASE_URL|ANTHROPIC_API_KEY.*PROXY" "$f" 2>/dev/null; then
        echo -e "${RED}[警告]${NC} $f 中发现代理相关环境变量，请手动检查并删除："
        grep -n -E "ANTHROPIC_AUTH_TOKEN|ANTHROPIC_BASE_URL|ANTHROPIC_API_KEY.*PROXY" "$f" | sed 's/^/    /'
        SHELL_CLEAN=false
    fi
done
if $SHELL_CLEAN; then
    echo -e "${GREEN}[OK]${NC} shell 配置文件无代理变量"
fi

echo ""

# ---- 5. 检查 CC Switch 进程 ----
if pgrep -f "cc-switch\|CC.Switch\|CCSwitch" >/dev/null 2>&1; then
    echo -e "${YELLOW}[提示]${NC} CC Switch 进程仍在运行，建议退出："
    echo "  可以在菜单栏右键退出，或执行："
    echo "  killall 'CC Switch' 2>/dev/null || killall cc-switch 2>/dev/null"
else
    echo -e "${GREEN}[OK]${NC} CC Switch 进程未运行"
fi

echo ""
echo "================================"
if [ $CHANGED -gt 0 ]; then
    echo -e "${GREEN}✅ 清理完成！共修改 $CHANGED 个文件${NC}"
else
    echo -e "${GREEN}✅ 所有配置已经是干净的，无需修改${NC}"
fi
echo ""
echo "现在可以运行 'claude' 启动 Claude Code，使用网页授权登录新账号。"
echo ""

    