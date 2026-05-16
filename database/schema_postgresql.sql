-- ============================================================
-- 校园安全 APP - PostgreSQL 数据库完整设计
-- 版本：PostgreSQL 15+
-- 编码：UTF-8
-- ============================================================

-- 创建数据库（手动执行一次）
-- CREATE DATABASE campus_safety ENCODING 'UTF8' LC_COLLATE='zh_CN.UTF-8';

-- 启用扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";   -- 支持模糊搜索
CREATE EXTENSION IF NOT EXISTS "unaccent";  -- 搜索去音调

-- ============================================================
-- 枚举类型
-- ============================================================
DO $$ BEGIN
  CREATE TYPE risk_level_enum  AS ENUM ('safe', 'medium', 'high');
  CREATE TYPE data_source_enum AS ENUM ('user_report', 'police', 'system', 'manual');
  CREATE TYPE platform_enum    AS ENUM ('ios', 'android');
  CREATE TYPE report_type_enum AS ENUM ('phone', 'sms', 'link', 'other');
  CREATE TYPE report_status    AS ENUM ('pending', 'approved', 'rejected');
  CREATE TYPE content_status   AS ENUM ('draft', 'published', 'archived');
  CREATE TYPE notif_type_enum  AS ENUM ('call_alert','sms_alert','fraud_alert','report_result','system');
  CREATE TYPE detect_type_enum AS ENUM ('incoming', 'manual_query');
  CREATE TYPE user_action_enum AS ENUM ('hung_up', 'answered', 'ignored', 'reported');
EXCEPTION WHEN duplicate_object THEN null;
END $$;

-- ============================================================
-- 1. 用户表
-- ============================================================
CREATE TABLE IF NOT EXISTS users (
  id               BIGSERIAL PRIMARY KEY,
  phone            VARCHAR(20)  NOT NULL UNIQUE,            -- 脱敏后的手机号
  phone_hash       CHAR(64)     NOT NULL UNIQUE,            -- SHA-256 哈希
  nickname         VARCHAR(50)  NOT NULL DEFAULT '校园守护者',
  school           VARCHAR(100),
  avatar_url       VARCHAR(255),

  blocked_calls    INTEGER      NOT NULL DEFAULT 0,
  alerted_sms      INTEGER      NOT NULL DEFAULT 0,
  total_reports    INTEGER      NOT NULL DEFAULT 0,
  cases_read       INTEGER      NOT NULL DEFAULT 0,
  protection_score SMALLINT     NOT NULL DEFAULT 0 CHECK (protection_score BETWEEN 0 AND 100),

  role             VARCHAR(20)  NOT NULL DEFAULT 'user',    -- user | admin
  status           SMALLINT     NOT NULL DEFAULT 1,         -- 1=正常 0=封禁
  last_login_at    TIMESTAMPTZ,
  created_at       TIMESTAMPTZ  NOT NULL DEFAULT now(),
  updated_at       TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX idx_users_school     ON users (school);
CREATE INDEX idx_users_created_at ON users (created_at DESC);

COMMENT ON TABLE  users              IS '用户账户表';
COMMENT ON COLUMN users.phone_hash   IS 'SHA-256 哈希，用于快速精确查找，不存明文';
COMMENT ON COLUMN users.protection_score IS '防护分 0-100，用于展示用户等级';

-- ============================================================
-- 2. 用户设备
-- ============================================================
CREATE TABLE IF NOT EXISTS user_devices (
  id          BIGSERIAL PRIMARY KEY,
  user_id     BIGINT       NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  device_id   VARCHAR(128) NOT NULL,
  platform    platform_enum NOT NULL,
  fcm_token   VARCHAR(256),                                 -- FCM / APNS Token
  app_version VARCHAR(20),
  os_version  VARCHAR(30),
  is_active   BOOLEAN      NOT NULL DEFAULT TRUE,
  created_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),
  updated_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),
  UNIQUE (user_id, device_id)
);

CREATE INDEX idx_user_devices_user ON user_devices (user_id);
COMMENT ON TABLE user_devices IS '用户设备注册，用于 FCM/APNs 推送';

-- ============================================================
-- 3. 诈骗电话号码库（核心）
-- ============================================================
CREATE TABLE IF NOT EXISTS fraud_phones (
  id               BIGSERIAL PRIMARY KEY,
  phone_number     VARCHAR(30)      NOT NULL,
  phone_hash       CHAR(64)         NOT NULL UNIQUE,

  risk_level       risk_level_enum  NOT NULL DEFAULT 'medium',
  risk_type        VARCHAR(50),
  risk_score       SMALLINT         NOT NULL DEFAULT 50 CHECK (risk_score BETWEEN 0 AND 100),

  source           data_source_enum NOT NULL DEFAULT 'user_report',
  location         VARCHAR(50),
  carrier          VARCHAR(30),

  report_count     INTEGER          NOT NULL DEFAULT 1,
  confirmed_count  INTEGER          NOT NULL DEFAULT 0,
  query_count      INTEGER          NOT NULL DEFAULT 0,

  is_verified      BOOLEAN          NOT NULL DEFAULT FALSE,
  is_active        BOOLEAN          NOT NULL DEFAULT TRUE,

  first_reported_at TIMESTAMPTZ     NOT NULL DEFAULT now(),
  last_reported_at  TIMESTAMPTZ     NOT NULL DEFAULT now()
);

CREATE INDEX idx_fraud_phones_risk_level   ON fraud_phones (risk_level);
CREATE INDEX idx_fraud_phone_score         ON fraud_phones (risk_score DESC);
CREATE INDEX idx_fraud_phones_risk_score   ON fraud_phones (risk_score DESC);
CREATE INDEX idx_fraud_phones_report_count ON fraud_phones (report_count DESC);
CREATE UNIQUE INDEX idx_fraud_phones_hash ON fraud_phones (phone_hash);
CREATE INDEX idx_fraud_phones_active       ON fraud_phones (is_active);

-- 全文搜索索引（号码前缀搜索）
CREATE INDEX idx_fraud_phones_trgm ON fraud_phones USING GIN (phone_number gin_trgm_ops);

COMMENT ON TABLE  fraud_phones              IS '诈骗电话号码库（核心表）';
COMMENT ON COLUMN fraud_phones.phone_hash   IS '加速精确查找，避免全表扫描';
COMMENT ON COLUMN fraud_phones.risk_score   IS '0-100，>=70高危，>=35中危，其余安全';

-- ============================================================
-- 4. 来电检测日志
-- ============================================================
CREATE TABLE IF NOT EXISTS call_detection_logs (
  id              BIGSERIAL PRIMARY KEY,
  user_id         BIGINT           NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  phone_number    VARCHAR(30)      NOT NULL,                -- 脱敏后
  fraud_phone_id  BIGINT           REFERENCES fraud_phones(id) ON DELETE SET NULL,
  risk_level      risk_level_enum  NOT NULL DEFAULT 'safe',
  detection_type  detect_type_enum NOT NULL DEFAULT 'manual_query',
  user_action     user_action_enum,
  detected_at     TIMESTAMPTZ      NOT NULL DEFAULT now()
);

CREATE INDEX idx_call_logs_user      ON call_detection_logs (user_id);
CREATE INDEX idx_call_logs_time      ON call_detection_logs (detected_at DESC);
CREATE INDEX idx_call_logs_risk      ON call_detection_logs (risk_level);

-- ============================================================
-- 5. 短信分析日志（不存原文）
-- ============================================================
CREATE TABLE IF NOT EXISTS sms_analysis_logs (
  id                 BIGSERIAL PRIMARY KEY,
  user_id            BIGINT      NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  sender             VARCHAR(50) NOT NULL,
  content_hash       CHAR(64)    NOT NULL,                  -- 原文哈希，原文留本地
  content_length     INTEGER     NOT NULL DEFAULT 0,
  risk_level         risk_level_enum NOT NULL DEFAULT 'safe',
  risk_score         SMALLINT    NOT NULL DEFAULT 0,
  matched_keywords   JSONB,                                 -- ["安全账户", "公安局"]
  has_suspicious_url BOOLEAN     NOT NULL DEFAULT FALSE,
  analyzed_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_sms_logs_user ON sms_analysis_logs (user_id);
CREATE INDEX idx_sms_logs_time ON sms_analysis_logs (analyzed_at DESC);
COMMENT ON COLUMN sms_analysis_logs.content_hash IS '短信原文的 SHA-256，原文不上传，保护隐私';

-- ============================================================
-- 6. 短信关键词规则库
-- ============================================================
CREATE TABLE IF NOT EXISTS sms_keywords (
  id          SERIAL PRIMARY KEY,
  keyword     VARCHAR(100) NOT NULL UNIQUE,
  risk_weight SMALLINT     NOT NULL DEFAULT 10 CHECK (risk_weight BETWEEN 1 AND 100),
  category    VARCHAR(50),
  is_active   BOOLEAN      NOT NULL DEFAULT TRUE,
  hit_count   INTEGER      NOT NULL DEFAULT 0,
  created_at  TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX idx_sms_keywords_category ON sms_keywords (category);
CREATE INDEX idx_sms_keywords_weight   ON sms_keywords (risk_weight DESC);
CREATE INDEX idx_sms_keywords_active   ON sms_keywords (is_active);

-- 全文搜索
CREATE INDEX idx_sms_keywords_trgm ON sms_keywords USING GIN (keyword gin_trgm_ops);
COMMENT ON TABLE sms_keywords IS '短信关键词风险规则库，risk_weight 越高风险越大';

-- ============================================================
-- 7. 诈骗案例库
-- ============================================================
CREATE TABLE IF NOT EXISTS fraud_cases (
  id           BIGSERIAL PRIMARY KEY,
  title        VARCHAR(200)     NOT NULL,
  summary      VARCHAR(500)     NOT NULL,
  content      TEXT             NOT NULL,                   -- Markdown 格式正文
  category     VARCHAR(20)      NOT NULL,
  risk_level   risk_level_enum  NOT NULL DEFAULT 'medium',
  emoji        VARCHAR(10)      NOT NULL DEFAULT '📋',
  view_count   INTEGER          NOT NULL DEFAULT 0,
  like_count   INTEGER          NOT NULL DEFAULT 0,
  share_count  INTEGER          NOT NULL DEFAULT 0,
  tags         JSONB,                                       -- ["刷单", "高校高发"]
  related_ids  JSONB,
  author_id    BIGINT,
  status       content_status   NOT NULL DEFAULT 'published',
  is_featured  BOOLEAN          NOT NULL DEFAULT FALSE,
  published_at TIMESTAMPTZ      NOT NULL DEFAULT now(),
  created_at   TIMESTAMPTZ      NOT NULL DEFAULT now(),
  updated_at   TIMESTAMPTZ      NOT NULL DEFAULT now()
);

CREATE INDEX idx_cases_category    ON fraud_cases (category);
CREATE INDEX idx_cases_risk        ON fraud_cases (risk_level);
CREATE INDEX idx_cases_view        ON fraud_cases (view_count DESC);
CREATE INDEX idx_cases_published   ON fraud_cases (published_at DESC);
CREATE INDEX idx_cases_featured    ON fraud_cases (is_featured);
CREATE INDEX idx_cases_status      ON fraud_cases (status);
CREATE INDEX idx_cases_tags        ON fraud_cases USING GIN (tags);

-- 全文搜索（中文需配合 pg_jieba 或 zhparser）
CREATE INDEX idx_cases_search ON fraud_cases USING GIN (to_tsvector('simple', title || ' ' || summary));

-- ============================================================
-- 8. 用户收藏案例
-- ============================================================
CREATE TABLE IF NOT EXISTS user_case_favorites (
  id         BIGSERIAL PRIMARY KEY,
  user_id    BIGINT      NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  case_id    BIGINT      NOT NULL REFERENCES fraud_cases(id) ON DELETE CASCADE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (user_id, case_id)
);

CREATE INDEX idx_favorites_user ON user_case_favorites (user_id);

-- ============================================================
-- 9. 电诈预警
-- ============================================================
CREATE TABLE IF NOT EXISTS fraud_alerts (
  id              BIGSERIAL PRIMARY KEY,
  title           VARCHAR(200)     NOT NULL,
  content         TEXT             NOT NULL,
  risk_level      risk_level_enum  NOT NULL DEFAULT 'medium',
  emoji           VARCHAR(10)      NOT NULL DEFAULT '📢',
  tags            JSONB,
  is_urgent       BOOLEAN          NOT NULL DEFAULT FALSE,
  push_count      INTEGER          NOT NULL DEFAULT 0,
  read_count      INTEGER          NOT NULL DEFAULT 0,
  report_count    INTEGER          NOT NULL DEFAULT 0,
  target_schools  JSONB,                                    -- NULL = 全量推送
  status          content_status   NOT NULL DEFAULT 'published',
  author_id       BIGINT,
  published_at    TIMESTAMPTZ      NOT NULL DEFAULT now(),
  created_at      TIMESTAMPTZ      NOT NULL DEFAULT now(),
  updated_at      TIMESTAMPTZ      NOT NULL DEFAULT now()
);

CREATE INDEX idx_alerts_risk      ON fraud_alerts (risk_level);
CREATE INDEX idx_alerts_urgent    ON fraud_alerts (is_urgent);
CREATE INDEX idx_alerts_published ON fraud_alerts (published_at DESC);
CREATE INDEX idx_alerts_status    ON fraud_alerts (status);

-- ============================================================
-- 10. 用户举报
-- ============================================================
CREATE TABLE IF NOT EXISTS user_reports (
  id              BIGSERIAL PRIMARY KEY,
  user_id         BIGINT           NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  report_type     report_type_enum NOT NULL,
  target          VARCHAR(500)     NOT NULL,
  target_hash     CHAR(64)         NOT NULL,
  description     TEXT,
  school          VARCHAR(100),
  status          report_status    NOT NULL DEFAULT 'pending',
  reviewer_id     BIGINT,
  review_note     VARCHAR(200),
  reviewed_at     TIMESTAMPTZ,
  fraud_phone_id  BIGINT           REFERENCES fraud_phones(id) ON DELETE SET NULL,
  ip_address      INET,                                     -- PostgreSQL 原生 IP 类型
  created_at      TIMESTAMPTZ      NOT NULL DEFAULT now()
);

CREATE INDEX idx_reports_user        ON user_reports (user_id);
CREATE INDEX idx_reports_status      ON user_reports (status);
CREATE INDEX idx_reports_target_hash ON user_reports (target_hash);
CREATE INDEX idx_reports_created     ON user_reports (created_at DESC);

-- ============================================================
-- 11. 推送通知日志
-- ============================================================
CREATE TABLE IF NOT EXISTS notification_logs (
  id       BIGSERIAL PRIMARY KEY,
  user_id  BIGINT           NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  type     notif_type_enum  NOT NULL,
  title    VARCHAR(200)     NOT NULL,
  body     TEXT             NOT NULL,
  data     JSONB,
  is_read  BOOLEAN          NOT NULL DEFAULT FALSE,
  sent_at  TIMESTAMPTZ      NOT NULL DEFAULT now(),
  read_at  TIMESTAMPTZ
);

CREATE INDEX idx_notif_user    ON notification_logs (user_id);
CREATE INDEX idx_notif_is_read ON notification_logs (is_read);
CREATE INDEX idx_notif_sent    ON notification_logs (sent_at DESC);

-- ============================================================
-- 12. 风险评估题库
-- ============================================================
CREATE TABLE IF NOT EXISTS quiz_questions (
  id          SERIAL PRIMARY KEY,
  question    TEXT       NOT NULL,
  options     JSONB      NOT NULL,  -- [{"label":"A","text":"...","is_correct":true}]
  explanation TEXT       NOT NULL,
  category    VARCHAR(50),
  difficulty  SMALLINT   NOT NULL DEFAULT 1 CHECK (difficulty BETWEEN 1 AND 3),
  is_active   BOOLEAN    NOT NULL DEFAULT TRUE,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_quiz_category   ON quiz_questions (category);
CREATE INDEX idx_quiz_difficulty ON quiz_questions (difficulty);

-- ============================================================
-- 13. 用户测试记录
-- ============================================================
CREATE TABLE IF NOT EXISTS user_quiz_records (
  id         BIGSERIAL PRIMARY KEY,
  user_id    BIGINT      NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  score      SMALLINT    NOT NULL,
  total      SMALLINT    NOT NULL,
  answers    JSONB       NOT NULL,
  weak_areas JSONB,
  taken_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_quiz_records_user ON user_quiz_records (user_id);

-- ============================================================
-- 触发器：自动更新 updated_at
-- ============================================================
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_users_updated       BEFORE UPDATE ON users       FOR EACH ROW EXECUTE FUNCTION set_updated_at();
CREATE TRIGGER trg_devices_updated     BEFORE UPDATE ON user_devices FOR EACH ROW EXECUTE FUNCTION set_updated_at();
CREATE TRIGGER trg_cases_updated       BEFORE UPDATE ON fraud_cases  FOR EACH ROW EXECUTE FUNCTION set_updated_at();
CREATE TRIGGER trg_alerts_updated      BEFORE UPDATE ON fraud_alerts FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ============================================================
-- 触发器：举报审核通过 → 自动写入诈骗号码库
-- ============================================================
CREATE OR REPLACE FUNCTION auto_approve_phone_report()
RETURNS TRIGGER AS $$
BEGIN
  IF NEW.status = 'approved' AND OLD.status = 'pending' AND NEW.report_type = 'phone' THEN
    INSERT INTO fraud_phones (phone_number, phone_hash, risk_level, source)
    VALUES (NEW.target, NEW.target_hash, 'high', 'user_report')
    ON CONFLICT (phone_hash) DO UPDATE
      SET report_count    = fraud_phones.report_count + 1,
          last_reported_at = now();

    UPDATE fraud_phones SET fraud_phones.id = id
    RETURNING id INTO NEW.fraud_phone_id;
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_auto_approve_report
  AFTER UPDATE OF status ON user_reports
  FOR EACH ROW EXECUTE FUNCTION auto_approve_phone_report();

-- ============================================================
-- 视图：用户防护等级
-- ============================================================
CREATE OR REPLACE VIEW v_user_protection AS
SELECT
  id,
  nickname,
  protection_score,
  CASE
    WHEN protection_score >= 80 THEN '防骗专家 🏆'
    WHEN protection_score >= 50 THEN '优秀守卫者 ⭐⭐⭐'
    WHEN protection_score >= 20 THEN '安全学员 ⭐⭐'
    ELSE                              '新手防护 ⭐'
  END AS protection_level,
  blocked_calls,
  alerted_sms,
  total_reports,
  cases_read
FROM users
WHERE status = 1;

-- ============================================================
-- 视图：高危电话排行（供管理后台）
-- ============================================================
CREATE OR REPLACE VIEW v_top_fraud_phones AS
SELECT
  id, phone_number, risk_level, risk_type,
  report_count, confirmed_count, query_count,
  location, is_verified, last_reported_at
FROM fraud_phones
WHERE is_active = TRUE
ORDER BY report_count DESC
LIMIT 100;

-- ============================================================
-- 初始化短信关键词数据
-- ============================================================
INSERT INTO sms_keywords (keyword, risk_weight, category) VALUES
  ('安全账户',    95, '金融'),
  ('公安局',      90, '冒充机构'),
  ('涉案资金',    92, '金融'),
  ('资产冻结',    88, '金融'),
  ('配合调查',    85, '冒充机构'),
  ('刷单',        85, '兼职'),
  ('刷好评',      80, '刷单'),
  ('解冻',        80, '金融'),
  ('冻结',        75, '金融'),
  ('立即转账',    90, '金融'),
  ('高额回报',    75, '投资'),
  ('点击链接',    70, '链接'),
  ('验证码',      70, '账号'),
  ('贷款',        60, '金融'),
  ('校园贷',      65, '校园'),
  ('助学贷款',    70, '校园'),
  ('内部名额',    65, '诱导'),
  ('佣金',        65, '兼职'),
  ('返现',        60, '购物'),
  ('代购',        55, '购物'),
  ('兼职',        45, '兼职'),
  ('私密',        50, '社交'),
  ('免费领取',    55, '诱导'),
  ('恭喜中奖',    80, '诱导'),
  ('账户异常',    75, '金融')
ON CONFLICT (keyword) DO NOTHING;

-- ============================================================
-- 初始化示例案例（供开发测试）
-- ============================================================
INSERT INTO fraud_cases (title, summary, content, category, risk_level, emoji, tags) VALUES
(
  '信用卡"解套"全套骗局实录',
  '冒充建行客服，要求将资金转至"安全账户"，共骗走28000元',
  E'## 案情经过\n\n小李是某高校大三学生，某天接到自称"建行客服"的电话...\n\n## 诈骗手法解析\n\n1. **权威身份包装**：冒充银行/公安，制造官方感\n2. **安全账户骗局**：世界上根本不存在"安全账户"\n\n## 防范要点\n\n- ✅ 任何"转账到安全账户"的要求都是诈骗\n- ✅ 挂断电话，主动拨打官方客服核实',
  '冒充客服', 'high', '💳', '["冒充银行", "安全账户", "高校高发"]'
),
(
  '兼职刷单骗局：先返利后骗款套路全解析',
  '以小额佣金建立信任，最终诱导大额垫付，单次损失最高数万元',
  E'## 案情经过\n\n大三学生小王在某平台接到"游戏代练"兼职邀请...\n\n## 防范要点\n\n- ✅ 任何需要垫付的兼职都是骗局\n- ✅ 正规平台不会要求先交保证金',
  '刷单诈骗', 'medium', '🎮', '["兼职诈骗", "游戏代练", "校园高发"]'
)
ON CONFLICT DO NOTHING;

-- 初始化预警数据
INSERT INTO fraud_alerts (title, content, risk_level, emoji, tags, is_urgent) VALUES
(
  '紧急预警：冒充辅导员收取培训费诈骗高发',
  '近期多所高校出现不法分子冒充辅导员发送短信，要求学生缴纳"职业技能培训费"，请务必通过官方渠道核实。此类诈骗专盯大一新生，已有多名同学上当。',
  'high', '🚨', '["高危", "校园场景", "新生专项"]', TRUE
)
ON CONFLICT DO NOTHING;
